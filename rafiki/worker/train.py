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
from rafiki.advisor import Proposal, ParamsType, TrainStrategy, EvalStrategy
from rafiki.model import BaseModel, load_model_class, serialize_knob_config, logger as model_logger
from rafiki.param_store import ParamStore

logger = logging.getLogger(__name__)

INVALID_TRIAL_SLEEP_SECS = 1
NO_NEXT_TRIAL_SLEEP_SECS = 5 * 60

class InvalidWorkerError(Exception): pass
class InvalidBudgetTypeError(Exception): pass

_JobInfo = namedtuple('_JobInfo', ['sub_train_job_id', 'sub_train_job_config', 'train_dataset_uri', 
                                'val_dataset_uri', 'model_id', 'model_file_bytes', 'model_class'])

class TrainWorker(object):
    def __init__(self, service_id, container_id, meta_store=None, param_store=None, **kwargs):
        self._job_monitor = _SubTrainJobMonitor(service_id, meta_store)
        self._params_root_dir = os.path.join(os.environ['WORKDIR_PATH'], os.environ['PARAMS_DIR_PATH'])
        self._worker_id = container_id
        self._redis_host = os.environ['REDIS_HOST']
        self._redis_port = os.environ['REDIS_PORT']
        self._client = make_superadmin_client()
        self._trial_id = None
        self._params_store: ParamStore = None

    def start(self):
        self._job_monitor.start()
        job_info = self._job_monitor.job_info
        sub_train_job_id = job_info.sub_train_job_id

        self._client.send_event('sub_train_job_worker_started', sub_train_job_id=sub_train_job_id)
        logger.info('Worker is for sub train job of ID "{}"'.format(sub_train_job_id))

        # Create params store
        self._params_store = ParamStore(session_id=sub_train_job_id, worker_id=self._worker_id, 
                                        redis_host=self._redis_host, redis_port=self._redis_port)

        # Load model class from bytes
        logger.info('Loading model class...')
        clazz = load_model_class(job_info.model_file_bytes, job_info.model_class)

        # Get Rafiki advior train worker to propose knobs in trials
        advisor_id = self._maybe_create_advisor(job_info, clazz)

        # Run model setup
        has_setup = False

        while True:
            # Secure a trial from store
            (self._trial_id, trial_no) = self._create_trial()
            if self._trial_id is None: # When there are no trials to conduct
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

                # Wait for a proposal from advisor for trial
                proposal = self._wait_for_proposal(advisor_id, trial_no, job_info)

                # Retrieve params for trial
                params = self._params_store.retrieve_params(proposal.params_type)

                # Train & evaluate model for trial
                logger.info('Running trial...')
                self._job_monitor.mark_trial_as_running(self._trial_id, proposal)
                (score, trial_params, params_dir) = \
                    self._train_and_evaluate_model(job_info, clazz, proposal, params)

                # Give feedback based on result of trial
                self._feedback(advisor_id, score, proposal)

                # Store output params of trial
                if trial_params is not None:
                    self._params_store.store_params(trial_params, score)

                # Mark trial as completed
                self._job_monitor.mark_trial_as_completed(self._trial_id, score, params_dir)
                
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

        # Train job must have finished, delete advisor & clear all params
        self._maybe_delete_advisor(advisor_id, job_info)
        self._params_store.clear_all_params()
            
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

    def _train_and_evaluate_model(self, job_info: _JobInfo, clazz: Type[BaseModel], proposal: Proposal, params):
        train_dataset_uri = job_info.train_dataset_uri
        val_dataset_uri = job_info.val_dataset_uri

        # Load model
        model_inst = clazz(train_strategy=proposal.train_strategy,
                            eval_strategy=proposal.eval_strategy,
                            **proposal.knobs)
        if params is not None:
            logger.info('Loading params for model...')
            model_inst.load_parameters(params)

        # Train model
        trial_params = None
        if proposal.train_strategy != TrainStrategy.NONE:
            logger.info('Training model...')
            model_inst.train(train_dataset_uri)
            trial_params = model_inst.dump_parameters() or None
            if trial_params:
                logger.info('Trial produced {} parameters'.format(len(trial_params)))

        # Evaluate model
        score = None
        if proposal.eval_strategy != EvalStrategy.NONE:
            logger.info('Evaluating model...')
            score = model_inst.evaluate(val_dataset_uri)
            logger.info('Trial score: {}'.format(score))

        # Save model
        params_dir = None
        if proposal.should_save_to_disk:
            logger.info('Saving trained model to disk...')
            params_dir = os.path.join(self._params_root_dir, self._trial_id)
            if not os.path.exists(params_dir):
                os.mkdir(params_dir)
            model_inst.save_parameters_to_disk(params_dir)

        return (score, trial_params, params_dir)

    def _create_trial(self):
        trial_no = None
        trial_id = None

        # Keep trying until worker creates a valid trial no & ID in store
        while trial_id is None:
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
                    break
            else:
                # Try to create trial with trial no
                # Trial ID might be returned as `None` if it fails
                trial_id = self._job_monitor.create_trial(trial_no, self._worker_id)
        
        return (trial_id, trial_no)

    def _wait_for_proposal(self, advisor_id, trial_no, job_info):
        proposal = None

        # Keep trying until a valid proposal, syncing trial statuses from store every try 
        while proposal is None:
            # Sync trials from underlying store
            self._job_monitor.sync_trials()

            # Determine total & concurrent trials
            (_, total_trials, concurrent_trial_nos) = self._job_monitor.get_sub_train_job_progress()
            logger.info('Trial nos concurrently running: {}'.format(concurrent_trial_nos))

            # Request proposal from advisor
            logger.info('Requesting for proposal from advisor...')
            proposal = self._client.get_proposal_from_advisor(advisor_id, self._worker_id, 
                                                            trial_no, total_trials, concurrent_trial_nos)

            if not proposal.is_valid:
                # Trial is still invalid
                proposal = None
                sleep_secs = INVALID_TRIAL_SLEEP_SECS
                logger.info('Trial #{} is currently invalid. Sleeping for {}s...'.format(trial_no, sleep_secs))
                time.sleep(sleep_secs)
                                                
        # Override knobs from sub train job config
        if 'knobs' in job_info.sub_train_job_config:
            override_knobs = job_info.sub_train_job_config['knobs']
            logger.info('Overriding proposal\'s knobs with {} from sub train job\'s config...'.format(override_knobs))
            proposal.knobs = { **proposal.knobs, **override_knobs }

        logger.info('Using proposal {}'.format(proposal.to_jsonable()))
        return proposal

    # Feedback result of trial to advisor, if score exists
    def _feedback(self, advisor_id, score, proposal: Proposal):
        if score is None:
            return
                    
        logger.info('Sending result of trial to advisor...')
        self._client.feedback_to_advisor(advisor_id, score, proposal)
        
    # Returns advisor ID to use
    def _maybe_create_advisor(self, job_info: _JobInfo, clazz):
        sub_train_job_config = job_info.sub_train_job_config

        # If user-configured advisor exists, use it
        if 'advisor_id' in sub_train_job_config:
            return sub_train_job_config['advisor_id']

        logger.info('Creating Rafiki advisor...')

        # Retrieve knob config for model of worker 
        knob_config = clazz.get_knob_config()

        # Create advisor associated with sub train job
        res = self._client.create_advisor(knob_config, advisor_id=job_info.sub_train_job_id)
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

class _SubTrainJobMonitor():
    '''
    Monitors & updates the status & trials of a sub train job for a given service ID.
    '''

    job_info: _JobInfo = None

    _seen_completed_trial_nos = set()
    _num_consec_completed_trials = 0
    _num_done_trials = 0
    _concurrent_trial_nos = []
    _total_trials = None

    def __init__(self, service_id: str, meta_store: MetaStore):
        self._meta_store = meta_store or MetaStore()
        self._service_id = service_id

    def start(self):
        service_id = self._service_id

        logger.info('Reading job info from meta store...')
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

        with self._meta_store:
            new_db_trials  = self._meta_store.get_trials_of_sub_train_job(sub_train_job_id, 
                                                                        min_trial_no=(self._num_consec_completed_trials + 1))
            num_done_trials = self._num_consec_completed_trials
            concurrent_trial_nos = []
            for x in new_db_trials:
                if x.status in [TrialStatus.COMPLETED, TrialStatus.ERRORED]:
                    num_done_trials += 1
                elif x.status in [TrialStatus.RUNNING, TrialStatus.PENDING]:
                    concurrent_trial_nos.append(x.no)

                if x.status == TrialStatus.COMPLETED and x.no not in self._seen_completed_trial_nos:
                    self._seen_completed_trial_nos.add(x.no)
            
            # Advance consecutive completed trial num
            while (self._num_consec_completed_trials + 1) in self._seen_completed_trial_nos:
                self._num_consec_completed_trials += 1 

            self._num_done_trials = num_done_trials
            self._concurrent_trial_nos = concurrent_trial_nos

        logger.info('Observed up to trial #{} completed'.format(self._num_consec_completed_trials))

    # Returns the progress of sub train job as (<next trial no>, <total trials>, <list of concurrently running trial nos>)
    # Returns <next trial no> as None if budget is reached
    # Should sync trials first
    def get_sub_train_job_progress(self):
        total_trials = self._total_trials
        num_valid_trials = self._num_done_trials + len(self._concurrent_trial_nos)
        next_trial_no = (num_valid_trials + 1) if num_valid_trials < total_trials else None
        return (next_trial_no, total_trials, self._concurrent_trial_nos)

    def create_trial(self, no, worker_id):
        sub_train_job_id = self.job_info.sub_train_job_id
        model_id = self.job_info.model_id

        try:
            with self._meta_store:
                trial = self._meta_store.create_trial(sub_train_job_id, no, model_id, worker_id)
                self._meta_store.commit()
                trial_id = trial.id

            logger.info('Created trial #{} of ID "{}" in store'.format(no, trial_id))
            return trial_id

        except DuplicateTrialNoError:
            logger.info('Avoided creating duplicate trial #{} in store!'.format(no))
            return None

    def mark_trial_as_errored(self, trial_id):
        logger.info('Marking trial as errored in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_errored(trial)
    
    def mark_trial_as_running(self, trial_id, proposal):
        logger.info('Marking trial as running in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_running(trial, proposal.to_jsonable())

    def mark_trial_as_completed(self, trial_id, score, params_dir):
        logger.info('Marking trial as completed in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_completed(trial, score, params_dir)

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
