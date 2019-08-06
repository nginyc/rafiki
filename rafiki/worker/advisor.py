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
from typing import Dict
import time
import traceback

from rafiki.utils.auth import superadmin_client
from rafiki.meta_store import MetaStore
from rafiki.model import load_model_class
from rafiki.advisor import make_advisor, BaseAdvisor
from rafiki.redis import TrainCache, ParamCache

LOOP_SLEEP_SECS = 0.1

class InvalidSubTrainJobError(Exception): pass

logger = logging.getLogger(__name__)

class _WorkerInfo():
    def __init__(self):
        self.trial_id: int = None # ID of pending trial assigned to worker, None if there is no pending trial

class AdvisorWorker():
    def __init__(self, service_id):
        self._monitor: _SubTrainJobMonitor = _SubTrainJobMonitor(service_id)
        self._redis_host = os.environ['REDIS_HOST']
        self._redis_port = os.environ['REDIS_PORT']
        self._train_cache: TrainCache = None
        self._param_cache: ParamCache = None
        self._advisor: BaseAdvisor = None
        self._worker_infos: Dict[str, _WorkerInfo] = {} # { <worker_id> : <info about worker> }

    def start(self):
        self._monitor.pull_job_info()
        self._train_cache = TrainCache(self._monitor.sub_train_job_id, 
                                        self._redis_host, 
                                        self._redis_port)
        self._param_cache = ParamCache(self._monitor.sub_train_job_id,
                                        self._redis_host,
                                        self._redis_port)
        self._advisor = self._make_advisor()

        logger.info(f'Starting advisor for sub train job "{self._monitor.sub_train_job_id}"...')
        self._notify_start()
        
        while True:
            self._fetch_results()
            if not self._make_proposals():
                self._notify_budget_reached()
                break
            time.sleep(LOOP_SLEEP_SECS)

    def stop(self):
        self._notify_stop()

        # Clear caches for sub train job
        try:
            self._train_cache.clear_all()
        except:
            logger.error('Error clearing train cache:')
            logger.error(traceback.format_exc())
        try:
            self._param_cache.clear_all_params()
        except:
            logger.error('Error clearing params cache:')
            logger.error(traceback.format_exc())


    def _notify_start(self):
        superadmin_client().send_event('sub_train_job_advisor_started', sub_train_job_id=self._monitor.sub_train_job_id)

    def _make_advisor(self):
        clazz = self._monitor.model_class
        budget = self._monitor.budget

        # Retrieve knob config
        knob_config = clazz.get_knob_config()
        advisor = make_advisor(knob_config, budget)
        logger.info(f'Using advisor "{type(advisor).__name__}"...')

        return advisor

    # Fetch results of workers
    def _fetch_results(self):
        for (worker_id, info) in self._worker_infos.items():
            # If no pending trial, skip
            if info.trial_id is None:
                continue

            # Fetch result for worker
            # If no result yet, skip
            result = self._train_cache.take_result(worker_id)
            if result is None:
                continue

            # Pass result to advisor
            self._advisor.feedback(worker_id, result)

            # Mark worker as not pending
            info.trial_id = None

    # Make proposals for workers
    # Returns False if tuning is to be stopped
    def _make_proposals(self):
        # For each free worker
        worker_ids = self._train_cache.get_workers()
        for worker_id in worker_ids:
            # If new worker, add info
            if worker_id not in self._worker_infos:
                self._worker_infos[worker_id] = _WorkerInfo()

            # Get info for worker
            worker_info = self._worker_infos[worker_id]

            # Check that worker doesn't already have a proposal
            proposal = self._train_cache.get_proposal(worker_id)
            if proposal is not None:
                continue

            # Create trial
            (trial_no, trial_id) = self._monitor.create_next_trial(worker_id)

            # Make proposal to free worker
            proposal = self._advisor.propose(worker_id, trial_no)

            # If advisor has no more proposals, to stop tuning
            if proposal is None:
                return False

            # Attach trial ID to proposal
            proposal.trial_id = trial_id
 
            # Push proposal to worker
            self._train_cache.create_proposal(worker_id, proposal)
            
            # Associate trial ID to worker
            worker_info.trial_id = trial_id
            
        return True

    def _notify_budget_reached(self):
        superadmin_client().send_event('sub_train_job_budget_reached', sub_train_job_id=self._monitor.sub_train_job_id)

    def _notify_stop(self):
        superadmin_client().send_event('sub_train_job_advisor_stopped', sub_train_job_id=self._monitor.sub_train_job_id)
        

class _SubTrainJobMonitor():
    '''
        Manages fetching & updating of metadata
    '''
    def __init__(self, service_id: str, meta_store: MetaStore = None):
        self.sub_train_job_id = None
        self.budget = None
        self.model_class = None
        self._num_trials = None
        self._meta_store = meta_store or MetaStore()
        self._service_id = service_id
        self._model_id = None

    def pull_job_info(self):
        service_id = self._service_id

        logger.info('Reading job info from meta store...')
        with self._meta_store:
            sub_train_job = self._meta_store.get_sub_train_job_by_advisor(service_id)
            if sub_train_job is None:
                raise InvalidSubTrainJobError('No sub train job associated with advisor "{}"'.format(service_id))

            train_job = self._meta_store.get_train_job(sub_train_job.train_job_id)
            if train_job is None:
                raise InvalidSubTrainJobError('No such train job with ID "{}"'.format(sub_train_job.train_job_id))

            model = self._meta_store.get_model(sub_train_job.model_id)
            if model is None:
                raise InvalidSubTrainJobError('No such model with ID "{}"'.format(sub_train_job.model_id))
            logger.info(f'Using model "{model.name}"...')
            logger.info(f'Using budget "{train_job.budget}"...')

            trials = self._meta_store.get_trials_of_sub_train_job(sub_train_job.id)

            self.sub_train_job_id = sub_train_job.id
            self.budget = train_job.budget
            self.model_class = load_model_class(model.model_file_bytes, model.model_class)
            self._num_trials = len(trials)
            self._model_id = model.id
            

    # Returns created trial number
    def create_next_trial(self, worker_id):
        self._num_trials += 1
        trial_no = self._num_trials

        with self._meta_store:
            trial = self._meta_store.create_trial(self.sub_train_job_id, trial_no, self._model_id, worker_id)
            self._meta_store.commit()
            trial_id = trial.id

            logger.info(f'Created trial #{trial_no} of ID "{trial_id}" in meta store')
            return (trial_no, trial_id)