#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import logging
import os
from typing import Union
import traceback
import time
from datetime import datetime

from rafiki.utils.auth import superadmin_client
from rafiki.meta_store import MetaStore
from rafiki.model import BaseModel, load_model_class, logger as model_logger
from rafiki.advisor import Proposal, TrialResult, ParamsType
from rafiki.redis import TrainCache, ParamCache
from rafiki.data_store import FileDataStore
from rafiki.param_store import FileParamStore, ParamStore

LOOP_SLEEP_SECS = 0.1
MAX_CONSEC_TRIAL_ERRORS = 100

class InvalidWorkerError(Exception): pass
class InvalidDatasetError(Exception): pass

logger = logging.getLogger(__name__)

class TrainWorker():
    def __init__(self, service_id, worker_id):
        self._worker_id = worker_id
        self._monitor: _SubTrainJobMonitor = _SubTrainJobMonitor(service_id)
        self._redis_host = os.environ['REDIS_HOST']
        self._redis_port = os.environ['REDIS_PORT']
        self._param_store: ParamStore = FileParamStore()
        self._trial_id = None # ID of currently running trial
        self._train_cache: TrainCache = None
        self._param_cache: ParamCache = None
        self._trial_errors = 0 # Consecutive traial errors

    def start(self):
        self._monitor.pull_job_info()
        self._train_cache = TrainCache(self._monitor.sub_train_job_id, 
                                        self._redis_host, 
                                        self._redis_port)
        self._param_cache = ParamCache(self._monitor.sub_train_job_id,
                                        self._redis_host,
                                        self._redis_port)

        logger.info(f'Starting worker for sub train job "{self._monitor.sub_train_job_id}"...')
        self._notify_start()
        
        while True:
            proposal = self._fetch_proposal()
            if proposal is not None:
                result = self._perform_trial(proposal)
                self._submit_result(result)
            time.sleep(LOOP_SLEEP_SECS)

    def stop(self):
        self._notify_stop()

        # If worker is currently running a trial, mark it has errored
        try:
            if self._trial_id is not None: 
                self._monitor.mark_trial_as_errored(self._trial_id)
        except:
            logger.error('Error marking trial as errored:')
            logger.error(traceback.format_exc())

        # Run model class teardown
        try:
            self._monitor.model_class.teardown()
        except:
            logger.error('Error tearing down model class:')
            logger.error(traceback.format_exc())

    def _notify_start(self):
        superadmin_client().send_event('train_job_worker_started', sub_train_job_id=self._monitor.sub_train_job_id)
        self._train_cache.add_worker(self._worker_id)

    def _fetch_proposal(self):
        proposal = self._train_cache.get_proposal(self._worker_id)
        return proposal

    def _perform_trial(self, proposal: Proposal) -> TrialResult:
        self._trial_id = proposal.trial_id

        logger.info(f'Starting trial {self._trial_id} with proposal {proposal}...')
        try:
            # Setup logging
            logger_info = self._start_logging_to_trial(
                    lambda log_line, log_lvl: self._monitor.log_to_trial(self._trial_id, log_line, log_lvl))

            self._monitor.mark_trial_as_running(self._trial_id, proposal)

            shared_params = self._pull_shared_params(proposal)
            model_inst = self._load_model(proposal)
            self._train_model(model_inst, proposal, shared_params)
            result = self._evaluate_model(model_inst, proposal)
            store_params_id = self._save_model(model_inst, proposal, result)
            model_inst.destroy()

            self._monitor.mark_trial_as_completed(self._trial_id, result.score, store_params_id)
            self._trial_errors = 0
            return result
        except Exception as e:
            logger.error('Error while running trial:')
            logger.error(traceback.format_exc())
            self._monitor.mark_trial_as_errored(self._trial_id)

            # Ensure that trial doesn't error too many times consecutively
            self._trial_errors += 1
            if self._trial_errors > MAX_CONSEC_TRIAL_ERRORS:
                logger.error(f'Reached {MAX_CONSEC_TRIAL_ERRORS} consecutive errors - raising exception')
                raise e

            return TrialResult(proposal)
        finally:
            self._stop_logging_to_trial(logger_info)

            # Untie from done trial 
            self._trial_id = None

    def _notify_stop(self):
        self._train_cache.delete_worker(self._worker_id)
        superadmin_client().send_event('train_job_worker_stopped', sub_train_job_id=self._monitor.sub_train_job_id)

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

    def _load_model(self, proposal: Proposal):
        logger.info('Creating model instance...')
        py_model_class = self._monitor.model_class
        model_inst = py_model_class(**proposal.knobs)
        return model_inst

    def _pull_shared_params(self, proposal: Proposal):
        if proposal.params_type == ParamsType.NONE:
            return None

        logger.info('Retrieving shared params from cache...')
        shared_params = self._param_cache.retrieve_params(proposal.params_type)
        return shared_params

    def _train_model(self, model_inst: BaseModel, proposal: Proposal, shared_params: Union[dict, None]):
        train_dataset_path = self._monitor.train_dataset_path
        train_args = self._monitor.train_args
        
        logger.info('Training model...')
        model_inst.train(train_dataset_path, shared_params=shared_params, **(train_args or {}))

    def _evaluate_model(self, model_inst: BaseModel, proposal: Proposal) -> TrialResult:
        val_dataset_path = self._monitor.val_dataset_path
        if not proposal.to_eval: 
            return TrialResult(proposal)
            
        logger.info('Evaluating model...')
        score = model_inst.evaluate(val_dataset_path)
        logger.info(f'Score on validation dataset: {score}')
        return TrialResult(proposal, score=score)
    
    def _save_model(self, model_inst: BaseModel, proposal: Proposal, result: TrialResult):
        if not proposal.to_cache_params and not proposal.to_save_params:
            return None
        
        logger.info('Dumping model parameters...')
        params = model_inst.dump_parameters()
        if proposal.to_cache_params:
            logger.info('Storing shared params in cache...')
            self._param_cache.store_params(params, score=result.score, time=datetime.now())
        
        store_params_id = None
        if proposal.to_save_params:
            logger.info('Saving shared params...')
            store_params_id = self._param_store.save(params)

        return store_params_id   

    def _submit_result(self, result: TrialResult):
        self._train_cache.create_result(self._worker_id, result)
        self._train_cache.delete_proposal(self._worker_id)

    def _stop_logging_to_trial(self, logger_info):
        (root_logger, py_model_logger, log_handler) = logger_info

        # Remove log handlers from loggers for this trial
        root_logger.removeHandler(log_handler)
        py_model_logger.removeHandler(log_handler)


class _SubTrainJobMonitor():
    '''
        Manages fetching & updating of metadata & datasets
    '''
    def __init__(self, service_id: str, meta_store: MetaStore = None):
        self.sub_train_job_id = None
        self.model_class = None
        self.train_dataset_path = None
        self.val_dataset_path = None
        self.train_args = None
        self._meta_store = meta_store or MetaStore()
        self._service_id = service_id
        self._data_store = FileDataStore()

    def pull_job_info(self):
        service_id = self._service_id

        logger.info('Reading job info from meta store...')
        with self._meta_store:
            worker = self._meta_store.get_train_job_worker(service_id)
            if worker is None:
                raise InvalidWorkerError('No such worker "{}"'.format(service_id))

            sub_train_job = self._meta_store.get_sub_train_job(worker.sub_train_job_id)
            if sub_train_job is None:
                raise InvalidWorkerError('No such sub train job associated with advisor "{}"'.format(service_id))

            train_job = self._meta_store.get_train_job(sub_train_job.train_job_id)
            if train_job is None:
                raise InvalidWorkerError('No such train job with ID "{}"'.format(sub_train_job.train_job_id))

            model = self._meta_store.get_model(sub_train_job.model_id)
            if model is None:
                raise InvalidWorkerError('No such model with ID "{}"'.format(sub_train_job.model_id))
            logger.info(f'Using model "{model.name}"...')

            (self.train_dataset_path, self.val_dataset_path) = self._load_datasets(train_job)
            self.train_args = train_job.train_args
            self.sub_train_job_id = sub_train_job.id
            self.model_class = load_model_class(model.model_file_bytes, model.model_class)

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

    def mark_trial_as_completed(self, trial_id, score, store_params_id):
        logger.info('Marking trial as completed in store...')
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.mark_trial_as_completed(trial, score, store_params_id)

    def log_to_trial(self, trial_id, log_line, log_lvl):
        with self._meta_store:
            trial = self._meta_store.get_trial(trial_id)
            self._meta_store.add_trial_log(trial, log_line, log_lvl)

    def _load_datasets(self, train_job):
        try:
            train_dataset = self._meta_store.get_dataset(train_job.train_dataset_id)
            assert train_dataset is not None
            val_dataset = self._meta_store.get_dataset(train_job.val_dataset_id)
            assert val_dataset is not None
            train_dataset_path = self._data_store.load(train_dataset.store_dataset_id)
            val_dataset_path = self._data_store.load(val_dataset.store_dataset_id)
            assert train_dataset_path is not None and val_dataset_path is not None
        except Exception as e:
            raise InvalidDatasetError(e)

        return (train_dataset_path, val_dataset_path)

class LoggerUtilsHandler(logging.Handler):
    def __init__(self, handle_log):
        logging.Handler.__init__(self)
        self._handle_log = handle_log

    def emit(self, record):
        log_line = str(record.msg)
        log_lvl = record.levelname
        self._handle_log(log_line, log_lvl)
