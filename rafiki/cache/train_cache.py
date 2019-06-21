from typing import Union, List
import logging

from rafiki.advisor import Proposal, ProposalResult
from .redis import RedisSession

logger = logging.getLogger(__name__)

REDIS_NAMESPACE = 'TRAIN'

class TrainCache(object):
    '''
    Caches proposals and proposal results to facilitates communication between advisor & train workers.

    For each session, assume a single advisor and multiple train workers running concurrently.

    :param str session_id: Associated session ID
    '''

    '''
        Internally, organises data into these Redis namespaces:

        workers:<worker_id>:proposal  | Proposal for worker
        workers:<worker_id>:result    | Result from worker
        workers                       | List of IDs of workers that are free
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

    def get_free_workers(self) -> List[str]:
        self._redis.acquire_lock() 
        try:
            worker_ids = self._redis.get('workers') or []
            return worker_ids
        finally:
            self._redis.release_lock() 

    def take_result(self, worker_id) -> Union[ProposalResult, None]:
        name = f'workers:{worker_id}:result'
        result = self._redis.get(name)
        if result is None:
            return None

        # Clear result from Redis
        self._redis.delete(name)
        return ProposalResult.from_jsonable(result)

    def create_proposal(self, worker_id: str, proposal: Proposal):
        name = f'workers:{worker_id}:proposal'
        assert self._redis.get(name) is None
        self._redis.set(name, proposal.to_jsonable())

    ####################################
    # Train Worker
    ####################################

    def add_free_worker(self, worker_id: str):
        self._redis.acquire_lock() 
        try:
            worker_ids = self._redis.get('workers') or []
            if worker_id not in worker_ids:
                worker_ids.append(worker_id)
            self._redis.set('workers', worker_ids)
        finally:
            self._redis.release_lock() 

    def take_proposal(self, worker_id: str) -> Union[Proposal, None]:
        name = f'workers:{worker_id}:proposal'
        proposal = self._redis.get(name)
        if proposal is None:
            return None

        # Clear proposal from Redis
        self._redis.delete(name)
        return Proposal.from_jsonable(proposal)

    def delete_free_worker(self, worker_id: str):
        self._redis.acquire_lock() 
        try:
            worker_ids = self._redis.get('workers') or []
            worker_ids = [x for x in worker_ids if x != worker_id]
            self._redis.set('workers', worker_ids)
        finally:
            self._redis.release_lock() 

    def create_result(self, result: ProposalResult):
        worker_id = result.worker_id
        name = f'workers:{worker_id}:result'
        assert self._redis.get(name) is None
        self._redis.set(name, result.to_jsonable())

