import time
import logging
import os
import traceback
import pickle
import pprint

from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.constants import TrainJobStatus, TrialStatus, BudgetType
from rafiki.model import load_model_class, serialize_knob_config, logger as model_logger
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
            
        self._service_id = service_id
        self._meta_store = meta_store
        self._param_store = param_store
        self._trial_id = None
        self._advisor_id = None
        self._client = self._make_client(admin_host, admin_port, advisor_host, advisor_port)

    def start(self):
        logger.info('Starting train worker for service of ID "{}"...' \
            .format(self._service_id))
            
        while True:
            (sub_train_job_id, budget, model_id, model_file_bytes, model_class, \
                train_job_id, train_dataset_uri, val_dataset_uri) = self._read_worker_info()

            # If budget reached, stop worker
            if self._if_budget_reached(budget, sub_train_job_id):
                logger.info('Budget for train job has reached')
                self._stop_worker()
    
                if self._advisor_id is not None:
                    self._delete_advisor(self._advisor_id)
                    self._advisor_id = None
                break

            # Otherwise, create a new trial
            self._trial_id = self._create_trial(model_id, sub_train_job_id)

            # Perform trial & record results
            try:
                logger.info('Starting trial...')

                # Load model class from bytes
                logger.info('Loading model class...')
                clazz = load_model_class(model_file_bytes, model_class)

                # If not created, create a Rafiki advisor for train worker to propose knobs in trials
                if self._advisor_id is None:
                    self._advisor_id = self._create_advisor(clazz)

                # Generate knobs for trial
                knobs = self._get_proposal_from_advisor()

                # Train & evaluate model for trial
                self._mark_trial_as_running(knobs)
                (score, parameters) = self._train_and_evaluate_model(clazz, knobs, train_dataset_uri, 
                                                                    val_dataset_uri)
                self._mark_trial_as_completed(score, parameters)

                # Report results of trial to advisor
                self._feedback_to_advisor(knobs, score)
            except Exception:
                self._mark_trial_as_errored()
                break # Exit worker upon trial error
            finally:
                # Untie from done trial 
                self._trial_id = None
            
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

    def _train_and_evaluate_model(self, clazz, knobs, train_dataset_uri, \
                                val_dataset_uri):
        logger.info('Training & evaluating model...')
        
        # Add log handlers for trial, including adding handler to root logger 
        # to capture any logs emitted with level above INFO during model training & evaluation
        def handle_log(log_line, log_lvl):
            with self._meta_store:
                trial = self._meta_store.get_trial(self._trial_id)
                self._meta_store.add_trial_log(trial, log_line, log_lvl)

        log_handler = ModelLoggerHandler(handle_log)
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
        model_inst.train(train_dataset_uri)

        # Evaluate model
        score = model_inst.evaluate(val_dataset_uri)

        # Dump and pickle model parameters
        parameters = model_inst.dump_parameters()
        model_inst.destroy()

        # Remove log handlers from loggers for this trial
        root_logger.removeHandler(log_handler)
        py_model_logger.removeHandler(log_handler)

        logger.info('Trial score: {}'.format(score))

        return (score, parameters)

    # Gets proposal of a set of knob values from advisor
    def _get_proposal_from_advisor(self):
        logger.info('Requesting for knobs proposal from advisor...')
        res = self._client.generate_proposal(self._advisor_id)
        knobs = res['knobs']
        logger.info('Received proposal of knobs from advisor:')
        logger.info(pprint.pformat(knobs))
        return knobs

    # Feedback result of knobs to advisor
    def _feedback_to_advisor(self, knobs, score):
        try:
            logger.info('Sending result of trials\' knobs to advisor...')
            self._client.feedback_to_advisor(self._advisor_id, knobs, score)
        except Exception:
            logger.error('Error while sending result of proposal to advisor:')
            logger.error(traceback.format_exc())

    def _stop_worker(self):
        logger.info('Stopping train job worker...')
        try:
            self._client.stop_train_job_worker(self._service_id)
        except Exception:
            # Throw just a warning - likely that another worker has stopped the service
            logger.warn('Error while stopping train job worker service:')
            logger.warn(traceback.format_exc())
        
    def _create_advisor(self, clazz):
        logger.info('Creating Rafiki advisor...')

        # Retrieve knob & train config for model of worker 
        knob_config = clazz.get_knob_config()
        knob_config_str = serialize_knob_config(knob_config)
        train_config = clazz.get_train_config()

        # Create advisor associated with worker
        advisor_type = train_config.get('advisor_type')
        res = self._client.create_advisor(knob_config_str, advisor_type, advisor_id=self._service_id)
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

    # Delete advisor
    def _delete_advisor(self, advisor_id):
        try:
            logger.info('Deleting advisor...')
            self._client.delete_advisor(advisor_id)
        except Exception:
            # Throw just a warning - not critical for advisor to be deleted
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
                train_job.budget,
                model.id,
                model.model_file_bytes,
                model.model_class,
                train_job.id,
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

    def _mark_trial_as_completed(self, score, parameters):
        logger.info('Storing model parameters...')
        parameters = pickle.dumps(parameters) # Convert to bytes
        param_id = self._trial_id
        self._param_store.put_params(param_id, parameters)

        logger.info('Marking trial as completed in DB...')
        with self._meta_store:
            trial = self._meta_store.get_trial(self._trial_id)
            self._meta_store.mark_trial_as_complete(trial, score, param_id)

    def _mark_trial_as_running(self, knobs):
        logger.info('Marking trial as running in DB...')
        with self._meta_store:
            trial = self._meta_store.get_trial(self._trial_id)
            self._meta_store.mark_trial_as_running(trial, knobs)

    def _make_client(self, admin_host, admin_port, advisor_host, advisor_port):
        superadmin_email = SUPERADMIN_EMAIL
        superadmin_password = SUPERADMIN_PASSWORD
        client = Client(admin_host=admin_host, 
                        admin_port=admin_port, 
                        advisor_host=advisor_host,
                        advisor_port=advisor_port)
        client.login(email=superadmin_email, password=superadmin_password)
        return client

class ModelLoggerHandler(logging.Handler):
    def __init__(self, handle_log):
        logging.Handler.__init__(self)
        self._handle_log = handle_log

    def emit(self, record):
        log_line = record.msg
        log_lvl = record.levelname
        self._handle_log(log_line, log_lvl)
      