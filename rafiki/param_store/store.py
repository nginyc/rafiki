import os
import json
import numpy as np
from typing import Dict
import uuid
import time
import logging
from itertools import chain
from collections import namedtuple
from datetime import datetime
import sys

from rafiki.model import Params
from rafiki.advisor import ParamsType

from .cache import Cache

logger = logging.getLogger(__name__)

class InvalidParamsError(Exception): pass
    
REDIS_NAMESPACE = 'PARAMS'
REDIS_LOCK_EXPIRE_SECONDS = 60
REDIS_LOCK_WAIT_SLEEP_SECONDS = 1 

_ParamMeta = namedtuple('_ParamMeta', ('param_id', 'score', 'time'))

class ParamStore(object):
    _worker_to_best: Dict[str, _ParamMeta] = {}
    _worker_to_recent: Dict[str, _ParamMeta] = {}

    '''
        Internally, organises data into these Redis namespaces:

        <session_id>:lock                | 0-1 lock for this session, 1 for acquired                
        <session_id>:param:<param_id>    | Params by ID
        <session_id>:meta                | Aggregate of all the required metadata: 
                                            { worker_to_best: { worker_id: { score, time, param_id } }, 
                                            worker_to_recent: { worker_id: { score, time, param_id } } }            
    '''
    
    '''
    Store API that retrieves and stores parameters for a session & a worker, backed an in-memory cache and Redis (optional).

    :param str session_id: Session ID associated with the parameters
    :param str worker_id: Worker ID associated with the parameters
    '''
    def __init__(self, session_id='local', worker_id='local', cache_size=4, redis_host=None, redis_port=6379):
        self._uid = str(uuid.uuid4()) # Process identifier for distributed locking
        self._sid = session_id
        self._worker_id = worker_id
        self._redis = self._make_redis_client(redis_host, redis_port) if redis_host is not None else None
        self._params_cache = Cache(cache_size)
        logger.info('Initializing params store of session "{}" for worker "{}"...'.format(session_id, worker_id))

    '''
    Stores parameters into underlying storage.

    :param Params params: Parameters as a { <name>: <numpy array> } dictionary
    :param datetime time: When the parameters were produced
    :param float score: Score associated with the parameters
    '''
    def store_params(self, params: Params, score: float = None, time: datetime = None):
        if params is None:
            raise InvalidParamsError('`params` cannot be `None`')   

        # Acquire lock to prevent race conditions
        if self._redis:
            self._acquire_redis_lock() 

        try:
            # With redis, sync in-memory metadata with Redis'
            if self._redis:
                (self._worker_to_best, self._worker_to_recent) = self._pull_metadata_from_redis()

            (param_ids_to_delete, param_meta) = self._update_metadata(score, time)
            
            if self._redis:
                if len(param_ids_to_delete) > 0:
                    # Delete outdated params from Redis
                    self._delete_params_from_redis(*param_ids_to_delete)
                
                if param_meta:
                    # Store input params on Redis
                    self._push_params_to_redis(param_meta.param_id, params)
                
                # Update metadata
                self._push_metadata_to_redis(self._worker_to_best, self._worker_to_recent)

            if param_meta:
                # Store input params in in-memory cache
                self._params_cache.put(param_meta.param_id, params)

        finally:
            # Release lock
            if self._redis:
                self._release_redis_lock()

    '''
    Retrieves parameters from underlying storage.

    :param ParamsType params_type: Type of parameters to retrieve
    :returns: Parameters as a { <name>: <numpy array> } dictionary
    :rtype: Params
    '''
    def retrieve_params(self, params_type: ParamsType) -> Params:
        # Acquire lock to prevent race conditions
        if self._redis:
            self._acquire_redis_lock()
            
        try:
            # With redis, sync in-memory metadata with Redis'
            if self._redis:
                (self._worker_to_best, self._worker_to_recent) = self._pull_metadata_from_redis()

            # Get param id to fetch
            param_id = self._get_params_by_type(params_type)
            if param_id is None:
                return None

            logger.info('To use params "{}"'.format(param_id))

            # Check in cache first
            params = self._params_cache.get(param_id)
            if params is not None:
                return params

            # Check in redis next, and store it in cache
            if self._redis:
                params = self._pull_params_from_redis(param_id)
                if params is None:
                    logger.error('Params don\'t exist in Redis!')
                    return None

                self._params_cache.put(param_id, params)
                return params
            
            return None

        finally:
            # Release lock
            if self._redis:
                self._release_redis_lock()

    '''
    Clears all parameters for this session from underlying storage.
    '''
    def clear_all_params(self):
        if self._redis:
            self._clear_all_from_redis()

    ####################################
    # Policies for params storage
    ####################################

    # Given input params with score & time, update metadata
    def _update_metadata(self, score: float, time: datetime):
        worker_id = self._worker_id

        # Record original param IDs
        og_param_ids = set([x.param_id for x in chain(self._worker_to_best.values(), self._worker_to_recent.values())])

        score = score or 0
        time = time or datetime.now()
        param_id = str(uuid.uuid4()) # Give it an ID
        param_meta = _ParamMeta(param_id, score, time)

        # Maintain best params
        prev_best_meta = self._worker_to_best.get(worker_id)
        if prev_best_meta is None or score > prev_best_meta.score:
            self._worker_to_best[worker_id] = param_meta

        # Maintain recent params
        prev_recent_meta = self._worker_to_recent.get(worker_id)
        if prev_recent_meta is None or time > prev_recent_meta.time:
            self._worker_to_recent[worker_id] = param_meta

        to_delete = [] # List of param IDs to be deleted

        # If any original param IDs are gone, to delete them
        # If input param ID was added, to store it
        new_param_ids = set([x.param_id for x in chain(self._worker_to_best.values(), self._worker_to_recent.values())])
        to_store = param_id in new_param_ids
        to_delete = [x for x in og_param_ids if x not in new_param_ids]

        return (to_delete, param_meta if to_store else None)

    def _get_params_by_type(self, params_type: ParamsType) -> str:
        if params_type == ParamsType.NONE:
            return None
        elif params_type == ParamsType.LOCAL_RECENT:
            return self._get_local_recent_params()
        elif params_type == ParamsType.LOCAL_BEST:
            return self._get_local_best_params()
        elif params_type == ParamsType.GLOBAL_RECENT:
            return self._get_global_recent_params()
        elif params_type == ParamsType.GLOBAL_BEST:
            return self._get_global_best_params()
        else:
            raise InvalidParamsError('No such params type: "{}"'.format(params_type))

    def _get_local_recent_params(self):
        worker_id = self._worker_id
        if worker_id not in self._worker_to_recent:
            return None
        
        params = self._worker_to_recent[worker_id]
        return params.param_id

    def _get_local_best_params(self):
        worker_id = self._worker_id
        if worker_id not in self._worker_to_best:
            return None
        
        params = self._worker_to_best[worker_id]
        return params.param_id

    def _get_global_recent_params(self):
        recent_params = [(params.time, params) for params in self._worker_to_recent.values()]
        if len(recent_params) == 0:
            return None

        recent_params.sort()
        (_, params) = recent_params[-1]
        return params.param_id

    def _get_global_best_params(self):
        best_params = [(params.score, params) for params in self._worker_to_best.values()]
        if len(best_params) == 0:
            return None

        best_params.sort()
        (_, params) = best_params[-1]
        return params.param_id

    ####################################
    # Redis communication
    ####################################

    def _pull_metadata_from_redis(self):  
        meta_name = self._get_redis_name('meta')
        metadata_str = self._get_from_redis(meta_name)

        worker_to_best = {}
        worker_to_recent = {}
        if metadata_str is not None:
            metadata = json.loads(metadata_str)
            logger.info('Pulled metadata from Redis: {}'.format(metadata))
            worker_to_best = { worker_id: self._jsonable_to_param_meta(jsonable) 
                            for (worker_id, jsonable) in metadata['worker_to_best'].items() }
            worker_to_recent = { worker_id: self._jsonable_to_param_meta(jsonable) 
                            for (worker_id, jsonable) in metadata['worker_to_recent'].items() }

        return (worker_to_best, worker_to_recent) 

    def _push_metadata_to_redis(self, worker_to_best, worker_to_recent):        
        meta_name = self._get_redis_name('meta')

        worker_to_best = { worker_id: self._param_meta_to_jsonable(meta) 
                        for (worker_id, meta) in worker_to_best.items() }
        worker_to_recent = { worker_id: self._param_meta_to_jsonable(meta) 
                        for (worker_id, meta) in worker_to_recent.items() }

        metadata = {
            'worker_to_best': worker_to_best,
            'worker_to_recent': worker_to_recent
        }

        logger.info('Pushing metadata to Redis: {}...'.format(metadata))

        metadata_str = json.dumps(metadata)
        self._redis.set(meta_name, metadata_str)

    def _delete_params_from_redis(self, *param_ids):
        logger.info('Deleting params: {}...'.format(param_ids))
        param_names = [self._get_redis_name('param:{}'.format(x)) for x in param_ids]
        self._redis.delete(*param_names)

    # Clears ALL metadata and params for session from Redis
    def _clear_all_from_redis(self):       
        params_name_patt = self._get_redis_name('*') 
        meta_name = self._get_redis_name('meta')
        params_keys = self._redis.keys(params_name_patt)
        if len(params_keys) > 0:
            logger.info('Clearing metadata and {} sets of params from Redis...'.format(len(params_keys)))
            self._redis.delete(meta_name, *params_keys)

    def _push_params_to_redis(self, param_id: str, params: Params):
        logger.info('Pushing params: "{}"...'.format(param_id))
        param_name = self._get_redis_name('param:{}'.format(param_id))
        params_str = self._serialize_params(params)
        self._redis.set(param_name, params_str)

    def _pull_params_from_redis(self, param_id: str) -> Params:
        logger.info('Pulling params: "{}"...'.format(param_id))
        param_name = self._get_redis_name('param:{}'.format(param_id))
        params_str = self._get_from_redis(param_name)
        if params_str is None:
            return None
            
        params = self._deserialize_params(params_str)
        return params

    def _make_redis_client(self, host, port):
        import redis
        cache_connection_url = 'redis://{}:{}'.format(host, port)
        connection_pool = redis.ConnectionPool.from_url(cache_connection_url)
        client = redis.StrictRedis(connection_pool=connection_pool, decode_responses=True)
        return client

    def _acquire_redis_lock(self):
        lock_value = self._uid        
        lock_name = self._get_redis_name('lock')

        # Keep trying to acquire lock
        res = None
        while not res:        
            res = self._redis.set(lock_name, lock_value, nx=True, ex=REDIS_LOCK_EXPIRE_SECONDS)
            if not res:
                sleep_secs = REDIS_LOCK_WAIT_SLEEP_SECONDS 
                logger.info('Waiting for lock to be released, sleeping for {}s...'.format(sleep_secs))
                time.sleep(sleep_secs)

        # Lock acquired
        logger.info('Acquired lock')

    def _release_redis_lock(self):
        lock_value = self._uid        
        lock_name = self._get_redis_name('lock')

        # Only release lock if it's confirmed to be the one I acquired
        # Possible that it was a lock acquired by someone else after my lock expired
        cur_lock_value = self._get_from_redis(lock_name)
        if cur_lock_value == lock_value: 
            self._redis.delete(lock_name)
            logger.info('Released lock')

    def _get_from_redis(self, key, value_type=str):
        value = self._redis.get(key)
        if value and value_type is str:
            return value.decode()
        return value
    
    def _param_meta_to_jsonable(self, param_meta: _ParamMeta):
        jsonable = param_meta._asdict()
        jsonable['time'] = str(jsonable['time'])
        return jsonable

    def _jsonable_to_param_meta(self, jsonable):
        jsonable['time'] = datetime.strptime(jsonable['time'], '%Y-%m-%d %H:%M:%S.%f')
        param_meta = _ParamMeta(**jsonable)
        return param_meta

    def _serialize_params(self, params):
        # Convert numpy arrays to lists
        params_for_json = { 
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for (name, value) in params.items() 
        }

        # Convert to JSON
        params_str = json.dumps(params_for_json)
        return params_str

    def _deserialize_params(self, params_str):
        # Convert from JSON
        params = json.loads(params_str)

        # Convert lists or numbers to numpy arrays
        for (name, value) in params.items():
            if isinstance(value, (list, int, float)):
                params[name] = np.asarray(value)
        
        return params

    def _get_redis_name(self, suffix):
        return '{}:{}:{}'.format(REDIS_NAMESPACE, self._sid, suffix)

