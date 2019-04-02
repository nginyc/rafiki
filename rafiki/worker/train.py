import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Type
from datetime import datetime

from rafiki.utils.auth import make_superadmin_client
from rafiki.client import Client
from rafiki.constants import BudgetType, TrainJobStatus, TrialStatus, ServiceStatus
from rafiki.meta_store import MetaStore, DuplicateTrialNoError
from rafiki.advisor import ParamsMonitor, Proposal, Params
from rafiki.model import BaseModel, load_model_class, serialize_knob_config, logger as model_logger
from rafiki.param_store import ParamStore

logger = logging.getLogger(__name__)

INVALID_TRIAL_SLEEP_SECS = 10
NO_NEXT_TRIAL_SLEEP_SECS = 5 * 60

class InvalidWorkerError(Exception): pass
class InvalidBudgetTypeError(Exception): pass

_JobInfo = namedtuple('_JobInfo', ['sub_train_job_id', 'sub_train_job_config', 'train_dataset_uri', 
                                'val_dataset_uri', 'model_id', 'model_file_bytes', 'model_class'])
_Trial = namedtuple('_Trial', ['id', 'no', 'datetime_started', 'out_shared_param_id', 'score', 'status', 'worker_id'])

class TrainWorker(object):
    def __init__(self, service_id, container_id, meta_store=None, param_store=None, **kwargs):
        self._job_monitor = _SubTrainJobMonitor(service_id, meta_store)
        self._params_root_dir = os.path.join(os.environ['WORKDIR_PATH'], os.environ['PARAMS_DIR_PATH'])
        self._worker_id = container_id
        self._param_store = param_store or ParamStore(redis_host=os.environ['REDIS_HOST'],
                                                        redis_port=os.environ['REDIS_PORT'])
        self._params_monitor = ParamsMonitor()
        self._client = make_superadmin_client()
        self._trial_id = None

    def start(self):
        self._job_monitor.start()
        job_info = self._job_monitor.job_info
        sub_train_job_id = job_info.sub_train_job_id

        self._client.send_event('sub_train_job_worker_started', sub_train_job_id=sub_train_job_id)
        logger.info('Worker is for sub train job of ID "{}"'.format(sub_train_job_id))

        # Load model class from bytes
        logger.info('Loading model class...')
        clazz = load_model_class(job_info.model_file_bytes, job_info.model_class)

        # Get Rafiki advior train worker to propose knobs in trials
        advisor_id = self._maybe_create_advisor(job_info, clazz)

        # Run model setup
        has_setup = False
        knobs = None

        while True:
            # Secure a trial from store
            trial_no = self._create_trial()
            if trial_no is None: # When there are no trials to conduct
                logger.info('Budget for sub train job has reached')
                self._client.send_event('sub_train_job_budget_reached', sub_train_job_id=sub_train_job_id)
                break

            # Perform trial & record results
            try:
                logger_info = self._start_logging_to_trial(
                    lambda log_line, log_lvl: 
                        self._job_monitor.log_to_trial(self._trial_id, log_line, log_lvl))

                # Setup model if not
                if not has_setup:
                    logger.info('Running model class setup...')
                    clazz.setup()
                    has_setup = True

                # Wait for trial to become valid
                trial_config = self._wait_for_trial_validity(trial_no, clazz)

                # Mark trial as started
                logger.info('Starting trial #{} of ID "{}"...'.format(trial_no, self._trial_id))
                self._job_monitor.mark_trial_as_started(self._trial_id)

                # Retrieve proposal
                knobs = self._retrieve_proposal(advisor_id, job_info, trial_config, knobs)
                
                # Generate knobs for trial
                knobs = self._retrieve_knobs(advisor_id, job_info, trial_config, knobs)

                # Generate shared params for trial
                (params, shared_param_id) = self._retrieve_params(job_info, trial_config)

                # Train & evaluate model for trial
                self._job_monitor.mark_trial_as_running(self._trial_id, knobs, shared_param_id)
                (score, trial_params, params_dir) = \
                    self._train_and_evaluate_model(job_info, clazz, trial_config, knobs, params)

                # If score exists, give feedback based on result of trial
                if score is not None:
                    knobs = self._feedback(advisor_id, job_info, score, knobs)

                # Store output shared params of trial
                trial_param_id = self._store_params(job_info, self._trial_id, trial_params)

                # Mark trial as completed
                self._job_monitor.mark_trial_as_completed(self._trial_id, score, params_dir, trial_param_id)
                
            except Exception as e:
                logger.error('Error while running trial:')
                logger.error(traceback.format_exc())
                self._job_monitor.mark_trial_as_errored(self._trial_id)
                raise e
            finally:
                self._stop_logging_to_trial(logger_info)

                # Untie from done trial 
                self._trial_id = None

        # Run model teardown
        if has_setup:
            logger.info('Running model class teardown...')
            clazz.teardown()

        # Train job must have finished, delete advisor & shared params
        self._maybe_delete_advisor(advisor_id, job_info)
        self._clear_params(job_info)
            
    def stop(self):
        job_info = self._job_monitor.job_info
        sub_train_job_id = job_info.sub_train_job_id if job_info is not None else None

        # If worker is currently running a trial, mark it has terminated
        try:
            if self._trial_id is not None: 
                self._job_monitor.mark_trial_as_terminated(self._trial_id)
        except Exception:
            logger.error('Error marking trial as terminated:')
            logger.error(traceback.format_exc())

        if sub_train_job_id is not None:
            self._client.send_event('sub_train_job_worker_stopped', sub_train_job_id=sub_train_job_id)

    def _start_logging_to_trial(self, handle_log):
        # Add log handlers for trial, including adding handler to root logger 
        # to capture any logs emitted with level above INFO during model training & evaluation
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

    def _train_and_evaluate_model(self, job_info: _JobInfo, clazz: Type[BaseModel],
                                knobs, params, should_save, should_evaluate):
        train_dataset_uri = job_info.train_dataset_uri
        val_dataset_uri = job_info.val_dataset_uri

        # Initialize & train model
        logger.info('Training model...')
        model_inst = clazz(**knobs)
        if len(params) > 0:
            model_inst.set_shared_parameters(params)
        model_inst.train(train_dataset_uri)
        trial_params = model_inst.get_shared_parameters() or None
        if trial_params:
            logger.info('Trial produced {} shared parameters'.format(len(trial_params)))

        # Evaluate model
        score = None
        if should_evaluate:
            logger.info('Evaluating model...')
            score = model_inst.evaluate(val_dataset_uri)
            logger.info('Trial score: {}'.format(score))

        # Save model
        params_dir = None
        if should_save:
            logger.info('Saving trained model...')
            params_dir = os.path.join(self._params_root_dir, self._trial_id)
            if not os.path.exists(params_dir):
                os.mkdir(params_dir)
            model_inst.save_parameters(params_dir)

        return (score, trial_params, params_dir)

    def _create_trial(self):
        trial_no = None

        while self._trial_id is None:
            # Sync trials from underlying store
            self._job_monitor.sync_trials()

            # Determine next trial no
            (trial_no, _, concurrent_trial_nos) = self._job_monitor.get_sub_train_job_progress()

            # If no next trial
            if trial_no is None:
                # If some trials as still running, sleep
                if len(concurrent_trial_nos) > 0:
                    sleep_secs = NO_NEXT_TRIAL_SLEEP_SECS
                    logger.info('Trial nos concurrently running: {}'.format(concurrent_trial_nos))
                    logger.info('No next trial but trials are still running. Sleeping for {}s...'.format(sleep_secs))
                    time.sleep(sleep_secs)
                
                # Otherwise, we're done with this sub train job
                else:
                    return None
            else:
                # Try to create trial with trial no
                self._trial_id = self._job_monitor.create_trial(trial_no, self._worker_id)
        
        return trial_no

    def _wait_for_trial_validity(self, trial_no, clazz: Type[BaseModel]):
        while True:
            # Update shared params monitor
            new_completed_trials = self._job_monitor.retrieve_new_completed_trials()
            for x in new_completed_trials:
                self._params_monitor.add_params(x.out_shared_param_id, x.score, 
                                                    x.datetime_started, x.worker_id)

            (_, total_trials, concurrent_trial_nos) = self._job_monitor.get_sub_train_job_progress()
            logger.info('Trial nos concurrently running: {}'.format(concurrent_trial_nos))

            # Check if trial is valid based on trial config
            trial_config = clazz.get_trial_config(trial_no, total_trials, concurrent_trial_nos)
            if trial_config.is_valid:
                # Good to start the trial
                return trial_config

            # Trial is still invalid
            sleep_secs = INVALID_TRIAL_SLEEP_SECS
            logger.info('Trial #{} is currently invalid. Sleeping for {}s...'.format(trial_no, sleep_secs))
            time.sleep(sleep_secs)

            # Sync trials from underlying store
            self._job_monitor.sync_trials()

    # Retrieves proposal of a set of knobs from advisor for a trial, overriding values as required
    def _retrieve_knobs(self, advisor_id, job_info: _JobInfo, trial_config: TrialConfig, knobs):
        if knobs is None:
            logger.info('Requesting for proposal from advisor...')
            res = self._client.generate_proposal(advisor_id)
            knobs = res['knobs']

        # Override knobs from trial config
        if len(trial_config.override_knobs) > 0:
            override_knobs = trial_config.override_knobs
            logger.info('Overriding knobs with {} from trial\'s config...'.format(override_knobs))
            knobs = { **knobs, **override_knobs }
            
        # Override knobs from sub train job config
        if 'knobs' in job_info.sub_train_job_config:
            override_knobs = job_info.sub_train_job_config['knobs']
            logger.info('Overriding knobs with {} from sub train job\'s config...'.format(override_knobs))
            knobs = { **knobs, **override_knobs }

        logger.info('Using knobs:')
        logger.info(knobs)

        return knobs

    # Retrieves shared params for a trial
    def _retrieve_params(self, job_info: _JobInfo, trial_config: TrialConfig):
        worker_id = self._worker_id

        # Get exact params
        param_id = self._params_monitor.get_params(trial_config.params, worker_id)

        # Load actual params from store
        params = {}
        if param_id is not None:
            logger.info('To use {} shared params'.format(trial_config.params.name))
            params = self._param_store.retrieve_params(job_info.sub_train_job_id, param_id)
            logger.info('Retrieved {} shared params'.format(len(params)))

        return (params, param_id)

    # Retrieves shared params for a trial
    def _store_params(self, job_info: _JobInfo, trial_id: str, trial_params: dict):
        sub_train_job_id = job_info.sub_train_job_id

        if trial_params is None:
            return

        logger.info('Storing shared params...')
        trial_param_id = self._param_store.store_params(sub_train_job_id, trial_params, trial_id)
        logger.info('Stored shared params of ID "{}"'.format(trial_param_id))

        return trial_param_id

    # Feedback result of trial to advisor
    def _feedback(self, advisor_id, job_info: _JobInfo, score, knobs):
        # Report results of trial to advisor and get next proposal of knobs
        logger.info('Sending result of trials\' knobs to advisor...')
        data = self._client.feedback_to_advisor(advisor_id, score, knobs)
        next_knobs = data['knobs']
        return next_knobs 
        
    # Returns advisor ID to use
    def _maybe_create_advisor(self, job_info: _JobInfo, clazz):
        sub_train_job_config = job_info.sub_train_job_config

        # If user-configured advisor exists, use it
        if 'advisor_id' in sub_train_job_config:
            return sub_train_job_config['advisor_id']

        logger.info('Creating Rafiki advisor...')

        # Retrieve knob & train config for model of worker 
        knob_config = clazz.get_knob_config()
        knob_config_str = serialize_knob_config(knob_config)

        # Create advisor associated with sub train job
        res = self._client.create_advisor(knob_config_str, advisor_id=job_info.sub_train_job_id)
        advisor_id = res['id']
        logger.info('Created advisor of ID "{}"'.format(advisor_id))

        return advisor_id

    def _maybe_delete_advisor(self, advisor_id, job_info: _JobInfo):
        sub_train_job_config = job_info.sub_train_job_config

        # If sub train job has user-configured advisor, don't delete
        if 'advisor_id' in sub_train_job_config and \
            sub_train_job_config['advisor_id'] == advisor_id:
            logger.info('Not deleting user-configured advisor...')
            return
            
        logger.info('Deleting advisor...')
        try:
            self._client.delete_advisor(advisor_id)
        except Exception:
            # Throw just a warning - maybe another worker deleted it
            logger.warning('Error while deleting advisor:')
            logger.warning(traceback.format_exc())

    def _clear_params(self, job_info: _JobInfo):
        sub_train_job_id = job_info.sub_train_job_id
        self._param_store.clear_params(sub_train_job_id)

class _SubTrainJobMonitor():
    '''
    Monitors & updates the status & trials of a sub train job for a given service ID.
    '''

    job_info: _JobInfo = None

    _trials = {} # { <id>: <trial> } 
    _new_completed_trials = []
    _total_trials = None
    _last_sync_datetime = None

    def __init__(self, service_id: str, meta_store: MetaStore):
        self._meta_store = meta_store or MetaStore()
        self._service_id = service_id

    def start(self):
        service_id = self._service_id

        logger.info('Reading job info from store...')
        with self._meta_store:
            worker = self._meta_store.get_sub_train_job_worker(service_id)
            if worker is None:
                raise InvalidWorkerError('No such worker with service ID "{}"'.format(service_id))

            sub_train_job = self._meta_store.get_sub_train_job(worker.sub_train_job_id)
            if sub_train_job is None:
                raise InvalidWorkerError('No such sub train job with ID "{}"'.format(worker.sub_train_job_id))

            train_job = self._meta_store.get_train_job(sub_train_job.train_job_id)
            if train_job is None:
                raise InvalidWorkerError('No such train job with ID "{}"'.format(sub_train_job.train_job_id))

            model = self._meta_store.get_model(sub_train_job.model_id)
            if model is None:
                raise InvalidWorkerError('No such model with ID "{}"'.format(sub_train_job.model_id))

            self.job_info = _JobInfo(sub_train_job.id, sub_train_job.config,
                                train_job.train_dataset_uri, train_job.val_dataset_uri,
                                model.id, model.model_file_bytes, model.model_class)
            self._total_trials = train_job.budget.get(BudgetType.MODEL_TRIAL_COUNT, 2)

    # Pulls new trials from store and updates internal record of trials & new trials
    def sync_trials(self):
        sub_train_job_id = self.job_info.sub_train_job_id
        last_sync_datetime = self._last_sync_datetime

        logger.info('Pulling new trials from store...')
        sync_datetime = datetime.now()
        with self._meta_store:
            if last_sync_datetime is None:
                new_db_trials = self._meta_store.get_trials_of_sub_train_job(sub_train_job_id)
            else:
                new_db_trials  = self._meta_store.get_trials_of_sub_train_job(sub_train_job_id, 
                                                                            min_datetime_updated=last_sync_datetime)
            
            # Update trials
            for x in new_db_trials:
                new_trial = _Trial(x.id, x.no, x.datetime_started, x.out_shared_param_id, 
                            x.score, x.status, x.worker_id)

                if x.id in self._trials:
                    existing_trial = self._trials[x.id]
                    
                    # Check if status changed
                    if new_trial.status != existing_trial.status:
                        logger.info('Trial #{} of ID "{}" changed from "{}" to "{}"'
                                .format(x.no, x.id, existing_trial.status, new_trial.status))
                        
                        if new_trial.status == TrialStatus.COMPLETED:
                            self._new_completed_trials.append(new_trial)
                        
                self._trials[x.id] = new_trial
                logger.info('Synced trial #{} of ID "{}"'.format(x.no, x.id))

        self._last_sync_datetime = sync_datetime

    def retrieve_new_completed_trials(self):
        trials = self._new_completed_trials
        self._new_completed_trials = []
        return trials

    # Returns the progress of sub train job as (<next trial no>, <total trials>, <list of concurrently running trial nos>)
    # Returns <next trial no> as None if budget is reached
    # Should sync trials first
    def get_sub_train_job_progress(self):
        total_trials = self._total_trials
        
        done_trials = [x for x in self._trials.values() if x.status in [TrialStatus.COMPLETED, TrialStatus.ERRORED]]
        concurrent_trials = [x for x in self._trials.values() if x.status in [TrialStatus.RUNNING, TrialStatus.STARTED]]
        valid_trials = done_trials + concurrent_trials

        next_trial_no = (len(valid_trials) + 1) if len(valid_trials) < total_trials else None
        concurrent_trial_nos = [x.no for x in concurrent_trials]
        
        return (next_trial_no, total_trials, concurrent_trial_nos)

    def create_trial(self, no, worker_id):
        sub_train_job_id = self.job_info.sub_train_job_id
        model_id = self.job_info.model_id

        try:
            with self._meta_store:
                logger.info('Creating new trial in store...')
                trial = self._meta_store.create_trial(sub_train_job_id, no, model_id, worker_id)
                self._meta_store.commit()
                logger.info('Created trial #{} of ID "{}" in store'.format(no, trial.id))
                return trial.id
        except DuplicateTrialNoError:
            logger.info('Avoided creating duplicate trial #{} in store!'.format(no))
            return None

    def mark_trial_as_errored(self, trial_id):
        logger.info('Marking trial as errored in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_errored(trial)
    
    def mark_trial_as_started(self, trial_id):
        logger.info('Marking trial as started in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_started(trial)

    def mark_trial_as_running(self, trial_id, knobs, shared_param_id):
        logger.info('Marking trial as running in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_running(trial, knobs, shared_param_id)

    def mark_trial_as_completed(self, trial_id, score, params_dir, out_shared_param_id):
        logger.info('Marking trial as completed in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_completed(trial, score, params_dir, out_shared_param_id)

    def mark_trial_as_terminated(self, trial_id):
        logger.info('Marking trial as terminated in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_terminated(trial)

    def log_to_trial(self, trial_id, log_line, log_lvl):
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.add_trial_log(trial, log_line, log_lvl)

class LoggerUtilsHandler(logging.Handler):
    def __init__(self, handle_log):
        logging.Handler.__init__(self)
        self._handle_log = handle_log

    def emit(self, record):
        log_line = str(record.msg)
        log_lvl = record.levelname
        self._handle_log(log_line, log_lvl)
