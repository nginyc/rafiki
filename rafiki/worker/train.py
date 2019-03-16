import logging
import os
import pickle
import pprint
import time
import traceback
from collections import namedtuple
from typing import Type

from rafiki.utils.auth import make_superadmin_client
from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.constants import BudgetType, TrainJobStatus, TrialStatus, ServiceStatus
from rafiki.meta_store import MetaStore
from rafiki.model import BaseModel, load_model_class, serialize_knob_config, logger as model_logger
from rafiki.param_store import ParamStore

logger = logging.getLogger(__name__)

class InvalidTrainJobError(Exception): pass
class InvalidSubTrainJobError(Exception): pass
class InvalidModelError(Exception): pass
class InvalidBudgetTypeError(Exception): pass

_SubTrainJob = namedtuple('_SubTrainJob', ['id', 'config'])
_TrainJob = namedtuple('_TrainJob', ['id', 'budget', 'train_dataset_uri', 'val_dataset_uri'])
_Model = namedtuple('_Model', ['id', 'model_file_bytes', 'model_class'])

class TrainWorker(object):
    def __init__(self, service_id, container_id, meta_store=None, param_store=None, **kwargs):
        if meta_store is None: 
            meta_store = MetaStore()
        
        if param_store is None: 
            param_store = ParamStore(redis_host=os.environ['REDIS_HOST'],
                                    redis_port=os.environ['REDIS_PORT'])
            
        self._params_root_dir = os.path.join(os.environ['WORKDIR_PATH'], os.environ['PARAMS_DIR_PATH'])
        self._service_id = service_id
        self._container_id = container_id
        self._meta_store = meta_store
        self._param_store = param_store
        self._trial_id = None
        self._client = make_superadmin_client()

    def start(self):
        (sub_train_job, train_job, model) = self._read_worker_info()
        self._sub_train_job_id = sub_train_job.id
        self._client.send_event('sub_train_job_worker_started', sub_train_job_id=self._sub_train_job_id)
        logger.info('Worker is for sub train job of ID "{}"'.format(sub_train_job.id))

        # Load model class from bytes
        logger.info('Loading model class...')
        clazz = load_model_class(model.model_file_bytes, model.model_class)

        # Get Rafiki advior train worker to propose knobs in trials
        advisor_id = self._maybe_create_advisor(sub_train_job, clazz)

        # Run model setup
        logger.info('Running model class setup...')
        clazz.setup()

        knobs = None
        params = None

        while True:
            # If budget reached, stop worker
            if self._if_budget_reached(train_job, sub_train_job):
                logger.info('Budget for sub train job has reached')
                self._client.send_event('sub_train_job_budget_reached', sub_train_job_id=self._sub_train_job_id)
                break

            # Otherwise, create a new trial
            self._trial_id = self._create_trial(sub_train_job, model)

            # Perform trial & record results
            try:
                logger.info('Starting trial...')

                logger_info = self._start_logging_to_trial()

                # Generate knobs for trial
                (knobs, params) = self._get_proposal(advisor_id, clazz, 
                                                    sub_train_job, knobs, params)

                # Train & evaluate model for trial
                self._mark_trial_as_running(knobs, params)
                (score, trial_params, params_dir) = \
                    self._train_and_evaluate_model(train_job, sub_train_job, clazz, knobs, params)

                # Give feedback based on result of trial
                (knobs, params) = self._feedback(advisor_id, sub_train_job, score, 
                                                knobs, trial_params)

                # Mark trial as completed
                self._mark_trial_as_completed(score, params_dir)
                
                self._stop_logging_to_trial(logger_info)

            except Exception as e:
                self._mark_trial_as_errored()
                raise e
            finally:
                # Untie from done trial 
                self._trial_id = None

        # Run model teardown
        logger.info('Running model class teardown...')
        clazz.teardown()

        # Train job must have finished, delete advisor & shared params
        self._maybe_delete_advisor(advisor_id, sub_train_job)
        self._clear_params(sub_train_job)
            
    def stop(self):
        # If worker is currently running a trial, mark it has terminated
        try:
            if self._trial_id is not None: 
                self._mark_trial_as_terminated()
        except Exception:
            logger.error('Error marking trial as terminated:')
            logger.error(traceback.format_exc())

        self._client.send_event('sub_train_job_worker_stopped', sub_train_job_id=self._sub_train_job_id)

    def _start_logging_to_trial(self):
        # Add log handlers for trial, including adding handler to root logger 
        # to capture any logs emitted with level above INFO during model training & evaluation
        def handle_log(log_line, log_lvl):
            with self._meta_store:
                trial = self._meta_store.get_trial(self._trial_id)
                self._meta_store.add_trial_log(trial, log_line, log_lvl)

        log_handler = LoggerUtilsHandler(handle_log)
        py_model_logger = logging.getLogger('{}.trial'.format(__name__))
        py_model_logger.setLevel(logging.INFO)
        py_model_logger.propagate = False # Avoid duplicate logs in root logger
        py_model_logger.addHandler(log_handler)
        model_logger.set_logger(py_model_logger)

        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        return (root_logger, py_model_logger, log_handler)

    def _stop_logging_to_trial(self, logger_info):
        (root_logger, py_model_logger, log_handler) = logger_info

        # Remove log handlers from loggers for this trial
        root_logger.removeHandler(log_handler)
        py_model_logger.removeHandler(log_handler)

    def _train_and_evaluate_model(self, train_job: _TrainJob, sub_train_job: _SubTrainJob, 
                                clazz: Type[BaseModel], knobs, params):
        # Initialize & train model
        logger.info('Training model...')
        model_inst = clazz(**knobs)
        params = model_inst.train(train_job.train_dataset_uri, params) or {}
        if len(params) > 0:
            logger.info('Trial produced {} shared parameters'.format(len(params)))

        # Evaluate model
        logger.info('Evaluating model...')
        score = model_inst.evaluate(train_job.val_dataset_uri)
        logger.info('Trial score: {}'.format(score))

        # Maybe save model
        params_dir = None
        if sub_train_job.config.get('should_save', True):
            logger.info('Saving model...')
            params_dir = os.path.join(self._params_root_dir, self._trial_id)
            model_inst.save_parameters(params_dir)

        return (score, params, params_dir)

    # Gets valid proposal of a set of knobs and params from advisor
    def _get_proposal(self, advisor_id, clazz: Type[BaseModel], sub_train_job, 
                    knobs, params):
        # Keep trying until valid proposal
        while True:
            if knobs is None or params is None:
                logger.info('Requesting for proposal from advisor...')
                res = self._client.generate_proposal(advisor_id, worker_id=self._container_id)
                knobs = res['knobs']
                params = res['params']

            # Validate knobs 
            validated_knobs = clazz.validate_knobs(knobs)
            if validated_knobs is None:
                logger.info('Knobs failed validation')
                (knobs, params) = self._feedback(advisor_id, sub_train_job, 0, knobs, {})
                continue
            
            knobs = validated_knobs
            break

        # Override knobs from sub train job config
        if 'knobs' in sub_train_job.config:
            config_knobs = sub_train_job.config['knobs']
            logger.info('Overriding knobs with {} from sub train job\'s config...'.format(config_knobs))
            knobs = { **knobs, **config_knobs }

        # Load actual params from store
        if len(params) > 0:
            logger.info('Retrieving shared params from store...')
            params = self._param_store.retrieve_params(sub_train_job.id, params)
        
        logger.info('Using proposal from advisor:')
        logger.info(knobs)
        if len(params) > 0:
            logger.info('With {} shared params'.format(len(params)))
            
        return (knobs, params)

    # Feedback result of trial to advisor
    def _feedback(self, advisor_id, sub_train_job: _SubTrainJob, score, knobs, trial_params):
        # Store shared params
        if len(trial_params) > 0:
            logger.info('Storing shared params to store...')
            trial_params = self._param_store.store_params(sub_train_job.id, trial_params, 
                                                            prefix=self._trial_id)

        # Report results of trial to advisor and get next proposal
        logger.info('Sending result of trials\' knobs & params to advisor...')
        data = self._client.feedback_to_advisor(advisor_id, score, knobs, 
                                                trial_params, worker_id=self._container_id)
        next_knobs = data['knobs']
        next_params = data['params']
        return (next_knobs, next_params) 
        
    # Returns advisor ID to use
    def _maybe_create_advisor(self, sub_train_job: _SubTrainJob, clazz):
        # If user-configured advisor exists, use it
        if 'advisor_id' in sub_train_job.config:
            return sub_train_job.config['advisor_id']

        logger.info('Creating Rafiki advisor...')

        # Retrieve knob & train config for model of worker 
        knob_config = clazz.get_knob_config()
        knob_config_str = serialize_knob_config(knob_config)

        # Create advisor associated with sub train job
        res = self._client.create_advisor(knob_config_str, advisor_id=sub_train_job.id)
        advisor_id = res['id']
        logger.info('Created advisor of ID "{}"'.format(advisor_id))

        return advisor_id

    def _create_trial(self, sub_train_job: _SubTrainJob, model: _Model):
        with self._meta_store:
            logger.info('Creating new trial in DB...')
            trial = self._meta_store.create_trial(
                sub_train_job_id=sub_train_job.id,
                model_id=model.id,
                worker_id=self._container_id
            )
            self._meta_store.commit()
            logger.info('Created trial of ID "{}" in DB'.format(trial.id))

            return trial.id

    def _maybe_delete_advisor(self, advisor_id, sub_train_job: _SubTrainJob):
        # If sub train job has user-configured advisor, don't delete
        if 'advisor_id' in sub_train_job.config and \
            sub_train_job.config['advisor_id'] == advisor_id:
            logger.info('Not deleting user-configured advisor...')
            return
            
        logger.info('Deleting advisor...')
        try:
            self._client.delete_advisor(advisor_id)
        except Exception:
            # Throw just a warning - maybe another worker deleted it
            logger.warning('Error while deleting advisor:')
            logger.warning(traceback.format_exc())

    def _clear_params(self, sub_train_job: _SubTrainJob):
        self._param_store.clear_params(sub_train_job.id)

    # Returns whether the worker reached its budget (only consider COMPLETED or ERRORED trials)
    def _if_budget_reached(self, train_job: _TrainJob, sub_train_job: _SubTrainJob):
        # By default, budget is model trial count of 2
        with self._meta_store:
            max_trials = train_job.budget.get(BudgetType.MODEL_TRIAL_COUNT, 2)
            trials = self._meta_store.get_trials_of_sub_train_job(sub_train_job.id)
            trials = [x for x in trials if x.status in [TrialStatus.COMPLETED, TrialStatus.ERRORED]]
            return len(trials) >= max_trials

    def _read_worker_info(self):
        logger.info('Reading info for worker...')
        with self._meta_store:
            sub_train_job = self._meta_store.get_sub_train_job_by_service(self._service_id)
            if sub_train_job is None:
                raise InvalidSubTrainJobError()

            train_job = self._meta_store.get_train_job(sub_train_job.train_job_id)
            if train_job is None:
                raise InvalidTrainJobError()

            model = self._meta_store.get_model(sub_train_job.model_id)
            if model is None:
                raise InvalidModelError()

            return (
                _SubTrainJob(sub_train_job.id, sub_train_job.config),
                _TrainJob(train_job.id, train_job.budget, train_job.train_dataset_uri, 
                        train_job.val_dataset_uri),
                _Model(model.id, model.model_file_bytes, model.model_class)
            )

    def _mark_trial_as_errored(self):
        logger.error('Error while running trial:')
        logger.error(traceback.format_exc())
        logger.info('Marking trial as errored in DB...')

        with self._meta_store:
            trial = self._meta_store.get_trial(self._trial_id)
            self._meta_store.mark_trial_as_errored(trial)

    def _mark_trial_as_completed(self, score, params_dir):
        logger.info('Marking trial as completed in DB...')
        with self._meta_store:
            trial = self._meta_store.get_trial(self._trial_id)
            self._meta_store.mark_trial_as_complete(trial, score, params_dir)

    def _mark_trial_as_running(self, knobs, params):
        logger.info('Marking trial as running in DB...')
        shared_params_count = len(params)
        with self._meta_store:
            trial = self._meta_store.get_trial(self._trial_id)
            self._meta_store.mark_trial_as_running(trial, knobs, shared_params_count)

    def _mark_trial_as_terminated(self):
        logger.info('Marking trial as terminated in DB...')
        with self._meta_store:
            trial = self._meta_store.get_trial(self._trial_id)
            self._meta_store.mark_trial_as_terminated(trial)

class LoggerUtilsHandler(logging.Handler):
    def __init__(self, handle_log):
        logging.Handler.__init__(self)
        self._handle_log = handle_log

    def emit(self, record):
        log_line = str(record.msg)
        log_lvl = record.levelname
        self._handle_log(log_line, log_lvl)
