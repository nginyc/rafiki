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

from typing import Union, List
import logging

from rafiki.advisor import Proposal, TrialResult
from .redis import RedisSession

logger = logging.getLogger(__name__)

REDIS_NAMESPACE = 'TRAIN'

class TrainCache(object):
    '''
    Caches proposals and trial results to facilitate communication between advisor & train workers.

    For each session, assume a single advisor and multiple train workers running concurrently.

    :param str session_id: Associated session ID
    '''

    '''
        Internally, organises data into these Redis namespaces:

        workers:<worker_id>:proposal  | Proposal for worker
        workers:<worker_id>:result    | Result from worker
        workers                       | Set of IDs of workers that are free
    '''
    
    def __init__(self, 
                session_id='local', 
                redis_host=None,
                redis_port=None):
        redis_namespace = f'{REDIS_NAMESPACE}:{session_id}'
        self._redis = RedisSession(redis_namespace, redis_host, redis_port)

    ####################################
    # Advisor
    ####################################

    def get_workers(self) -> List[str]:
        worker_ids = self._redis.list_set('workers') or []
        return worker_ids

    def take_result(self, worker_id) -> Union[TrialResult, None]:
        name = f'workers:{worker_id}:result'
        result = self._redis.get(name)
        if result is None:
            return None

        # Clear result from Redis
        self._redis.delete(name)
        logger.info(f'Retrieved result "{result}" for worker "{worker_id}"')
        return TrialResult.from_jsonable(result)

    def get_proposal(self, worker_id: str) -> Union[Proposal, None]:
        name = f'workers:{worker_id}:proposal'
        proposal = self._redis.get(name)
        if proposal is None:
            return None
        proposal = Proposal.from_jsonable(proposal)
        return proposal

    def create_proposal(self, worker_id: str, proposal: Proposal):
        name = f'workers:{worker_id}:proposal'
        assert self._redis.get(name) is None
        logger.info(f'Creating proposal "{proposal}" for worker "{worker_id}"...')
        self._redis.set(name, proposal.to_jsonable())

    def clear_all(self):
        logger.info(f'Clearing proposals & trial results...')
        self._redis.delete('workers')
        self._redis.delete_pattern('workers:*')

    ####################################
    # Train Worker
    ####################################

    def add_worker(self, worker_id: str):
        self._redis.add_to_set('workers', worker_id)

    def delete_proposal(self, worker_id: str):
        name = f'workers:{worker_id}:proposal'
        logger.info(f'Deleting existing proposal for worker "{worker_id}"...')
        self._redis.delete(name)

    def delete_worker(self, worker_id: str):
        self._redis.delete_from_set('workers', worker_id)

    def create_result(self, worker_id: str, result: TrialResult):
        name = f'workers:{worker_id}:result'
        assert self._redis.get(name) is None
        logger.info(f'Creating result "{result}" for worker "{worker_id}"...')
        self._redis.set(name, result.to_jsonable())

