import time
import logging
import os
import traceback
import pickle
import pprint

from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.constants import TrainJobStatus, TrialStatus, BudgetType
from rafiki.advisor import serialize_knob_config
from rafiki.model import load_model_class, logger as model_logger, utils as model_utils
from rafiki.meta_store import MetaStore
from rafiki.param_store import ParamStore
from rafiki.client import Client

logger = logging.getLogger(__name__)

class InvalidTrainJobException(Exception): pass
class InvalidModelException(Exception): pass
class InvalidBudgetTypeException(Exception): pass
class InvalidWorkerException(Exception): pass

class TrainWorker(object):
    def __init__(self, service_id, meta_store=None, param_store=None, **kwargs):
        if meta_store is None: 
            meta_store = MetaStore()
        
        if param_store is None: 
            param_store = ParamStore()

        admin_host = kwargs.get('admin_host', os.environ['ADMIN_HOST'])
        admin_port = kwargs.get('admin_port', os.environ['ADMIN_PORT'])
        advisor_host = kwargs.get('advisor_host', os.environ['ADVISOR_HOST'])
        advisor_port = kwargs.get('advisor_port', os.environ['ADVISOR_PORT'])
            
        self._params_root_dir = os.path.join(os.environ['WORKDIR_PATH'], os.environ['PARAMS_DIR_PATH'])
        self._service_id = service_id
        self._meta_store = meta_store
        self._param_store = param_store
        self._trial_id = None
        self._client = self._make_client(admin_host, admin_port, advisor_host, advisor_port)

    def start(self):
        logger.info('Reading info for worker...')
        (sub_train_job_id, sub_train_job_config, advisor_id, budget, 
            model_id, model_file_bytes, model_class, 
            train_dataset_uri, val_dataset_uri) = self._read_worker_info()
        logger.info('Worker is for sub train job of ID "{}"'.format(sub_train_job_id))

        # Load model class from bytes
        logger.info('Loading model class...')
        clazz = load_model_class(model_file_bytes, model_class)

        # Get Rafiki advior train worker to propose knobs in trials
        advisor_id = self._maybe_create_advisor(advisor_id, sub_train_job_id,
                                                sub_train_job_config, clazz)

        # Run model setup
        logger.info('Running model class setup...')
        clazz.setup()

        # Generate first knobs for trial
        (knobs, params) = self._get_proposal_from_advisor(advisor_id)

        while True:
            # If budget reached, stop worker
            if self._if_budget_reached(budget, sub_train_job_id):
                logger.info('Budget for sub train job has reached')
                self._maybe_delete_advisor(sub_train_job_config, advisor_id)
                self._stop_sub_train_job(sub_train_job_id)
                break

            # Otherwise, create a new trial
            self._trial_id = self._create_trial(model_id, sub_train_job_id)

            # Perform trial & record results
            try:
                logger.info('Starting trial...')

                # Load shared params
                params = self._param_store.retrieve_params(sub_train_job_id, params)

                # Train & evaluate model for trial
                self._mark_trial_as_running(knobs, params)
                (model_inst, score) = self._train_and_evaluate_model(clazz, knobs, params, train_dataset_uri, 
                                                                    val_dataset_uri)

                # Save parameters for trial
                (trial_params_dir, trial_params) = self._maybe_save_model(model_inst, sub_train_job_config)

                # Mark trial as completed
                self._mark_trial_as_completed(score, trial_params_dir)

                # Store shared params
                trial_params = self._param_store.store_params(sub_train_job_id, trial_params, 
                                                               prefix=self._trial_id)

                # Report results of trial to advisor and get next proposal
                (knobs, params) = self._feedback_to_advisor(advisor_id, score, knobs, trial_params)

            except Exception:
                self._mark_trial_as_errored()
                break # Exit worker upon trial error
            finally:
                # Untie from done trial 
                self._trial_id = None

        # Run model teardown
        logger.info('Running model class teardown...')
        clazz.teardown()
            
    def stop(self):
        # If worker is currently running a trial, mark it has terminated
        logger.info('Marking trial as terminated in DB...')
        try:
            if self._trial_id is not None: 
                with self._meta_store:
                    trial = self._meta_store.get_trial(self._trial_id)
                    self._meta_store.mark_trial_as_terminated(trial)

        except Exception:
            logger.error('Error marking trial as terminated:')
            logger.error(traceback.format_exc())

    def _train_and_evaluate_model(self, clazz, knobs, params, train_dataset_uri, 
                                val_dataset_uri):
        logger.info('Training & evaluating model...')

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

        # Initialize model
        model_inst = clazz(**knobs)

        # Train model
        model_inst.train(train_dataset_uri, params)

        # Evaluate model
        score = model_inst.evaluate(val_dataset_uri)

        # Remove log handlers from loggers for this trial
        root_logger.removeHandler(log_handler)
        py_model_logger.removeHandler(log_handler)

        logger.info('Trial score: {}'.format(score))

        return (model_inst, score)

    def _maybe_save_model(self, model_inst, sub_train_job_config):
        # Save model parameters
        params_dir = None
        if sub_train_job_config.get('should_save', True):
            params_dir = os.path.join(self._params_root_dir, self._trial_id)
            model_inst.save_parameters(params_dir)

        # Get shared params
        params = model_inst.get_shared_parameters()

        return (params_dir, params)

    # Gets proposal of a set of knobs and params from advisor
    def _get_proposal_from_advisor(self, advisor_id):
        logger.info('Requesting for proposal from advisor...')
        res = self._client.generate_proposal(advisor_id)
        knobs = res['knobs']
        params = res['params']
        logger.info('Received proposal from advisor:')
        logger.info(pprint.pformat(knobs))
        logger.info('With {} params'.format(len(params)))
        return (knobs, params)

    # Feedback result of knobs to advisor
    def _feedback_to_advisor(self, advisor_id, score, knobs, params):
        logger.info('Sending result of trials\' knobs & params to advisor...')
        return self._client.feedback_to_advisor(advisor_id, score, knobs, params)

    def _stop_sub_train_job(self, sub_train_job_id):
        logger.info('Stopping sub train job...')
        try:
            self._client.stop_sub_train_job(sub_train_job_id)
        except Exception:
            # Throw just a warning - likely that another worker has stopped it
            logger.warn('Error while stopping sub train job:')
            logger.warn(traceback.format_exc())
        
    # Returns advisor ID to use
    def _maybe_create_advisor(self, advisor_id, sub_train_job_id, sub_train_job_config, clazz):
        # If advisor already exists, use it
        if advisor_id is not None:
            return advisor_id

        # If user-configured advisor exists, use it
        if 'advisor_id' in sub_train_job_config:
            return sub_train_job_config['advisor_id']

        logger.info('Creating Rafiki advisor...')

        # Retrieve knob & train config for model of worker 
        knob_config = clazz.get_knob_config()
        knob_config_str = serialize_knob_config(knob_config)

        # Create advisor associated with sub train job
        res = self._client.create_advisor(knob_config_str, advisor_id=sub_train_job_id)
        advisor_id = res['id']
        logger.info('Created advisor of ID "{}"'.format(advisor_id))

        return advisor_id

    def _create_trial(self, model_id, sub_train_job_id):
        with self._meta_store:
            logger.info('Creating new trial in DB...')
            trial = self._meta_store.create_trial(
                sub_train_job_id=sub_train_job_id,
                model_id=model_id
            )
            self._meta_store.commit()
            logger.info('Created trial of ID "{}" in DB'.format(trial.id))

            return trial.id

    def _maybe_delete_advisor(self, sub_train_job_config, advisor_id):
        # If sub train job has user-configured advisor, don't delete
        if 'advisor_id' in sub_train_job_config and \
            sub_train_job_config['advisor_id'] == advisor_id:
            logger.info('Not deleting user-configured advisor of ID "{}"...' \
                .format(sub_train_job_config['advisor_id']))
            
        try:
            logger.info('Deleting advisor...')
            self._client.delete_advisor(advisor_id)
        except Exception:
            # Throw just a warning - maybe another worker deleted it
            logger.warning('Error while deleting advisor:')
            logger.warning(traceback.format_exc())

    # Returns whether the worker reached its budget (only consider COMPLETED or ERRORED trials)
    def _if_budget_reached(self, budget, sub_train_job_id):
        # By default, budget is model trial count of 2
        with self._meta_store:
            max_trials = budget.get(BudgetType.MODEL_TRIAL_COUNT, 2)
            trials = self._meta_store.get_trials_of_sub_train_job(sub_train_job_id)
            trials = [x for x in trials if x.status in [TrialStatus.COMPLETED, TrialStatus.ERRORED]]
            return len(trials) >= max_trials

    def _read_worker_info(self):
        with self._meta_store:
            worker = self._meta_store.get_train_job_worker(self._service_id)

            if worker is None:
                raise InvalidWorkerException()

            train_job = self._meta_store.get_train_job(worker.train_job_id)
            sub_train_job = self._meta_store.get_sub_train_job(worker.sub_train_job_id)
            model = self._meta_store.get_model(sub_train_job.model_id)

            if model is None:
                raise InvalidModelException()

            if train_job is None or sub_train_job is None:
                raise InvalidTrainJobException()

            return (
                sub_train_job.id,
                sub_train_job.config,
                sub_train_job.advisor_id,
                train_job.budget,
                model.id,
                model.model_file_bytes,
                model.model_class,
                train_job.train_dataset_uri,
                train_job.val_dataset_uri
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

    def _make_client(self, admin_host, admin_port, advisor_host, advisor_port):
        superadmin_email = SUPERADMIN_EMAIL
        superadmin_password = SUPERADMIN_PASSWORD
        client = Client(admin_host=admin_host, 
                        admin_port=admin_port, 
                        advisor_host=advisor_host,
                        advisor_port=advisor_port)
        client.login(email=superadmin_email, password=superadmin_password)
        return client

class LoggerUtilsHandler(logging.Handler):
    def __init__(self, handle_log):
        logging.Handler.__init__(self)
        self._handle_log = handle_log

    def emit(self, record):
        log_line = record.msg
        log_lvl = record.levelname
        self._handle_log(log_line, log_lvl)
      