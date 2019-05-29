import os
import json
import numpy as np
from typing import Dict
import uuid
import time
import logging
from collections import namedtuple
from datetime import datetime
import sys
from enum import Enum

from rafiki.model import Params

from .cache import Cache

logger = logging.getLogger(__name__)

class InvalidParamsError(Exception): pass

class ParamsType(Enum):
    LOCAL_RECENT = 'LOCAL_RECENT'
    LOCAL_BEST = 'LOCAL_BEST'
    GLOBAL_RECENT = 'GLOBAL_RECENT'
    GLOBAL_BEST = 'GLOBAL_BEST'
    NONE = 'NONE'
    
REDIS_NAMESPACE = 'PARAMS'
REDIS_LOCK_EXPIRE_SECONDS = 60
REDIS_LOCK_WAIT_SLEEP_SECONDS = 0.1

_ParamMeta = namedtuple('_ParamMeta', ('param_id', 'score', 'time'))

class ParamStore(object):
    '''
        Internally, organises data into these Redis namespaces:

        <session_id>:lock                | 0-1 lock for this session, 1 for acquired                
        <session_id>:param:<param_id>    | Params by ID
        <session_id>:meta                | Aggregate of all global metadata: 
                                            { params:   { GLOBAL_BEST:     { score, time, param_id }, 
                                                        { GLOBAL_RECENT:   { score, time, param_id }   }            
    '''
    
    '''
    Store API that retrieves and stores parameters for a session & a worker, backed by an in-memory cache and Redis for cross-worker sharing (optional).

    :param str session_id: Session ID associated with the parameters
    :param str worker_id: Worker ID associated with the parameters
    '''
    def __init__(self, session_id='local', worker_id='local', cache_size=4, redis_host=None, redis_port=6379):
        self._uid = str(uuid.uuid4()) # Process identifier for distributed locking
        self._params: Dict[str, _ParamMeta] = {} # Stores params metadata
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
                self._pull_from_redis()

            param_meta = self._update_params_meta(score, time)
            if param_meta:
                # Store input params in in-memory cache
                self._params_cache.put(param_meta.param_id, params)

            if self._redis:
                self._push_to_redis()

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
                self._pull_from_redis()

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

    # Given input params with score & time, update params metadata
    # Returns param meta for the input params, None if params meta is not to be stored
    def _update_params_meta(self, score: float, time: datetime):
        score = score or 0
        time = time or datetime.now()
        param_id = str(uuid.uuid4()) # Give it an ID
        param_meta = _ParamMeta(param_id, score, time)
        
        # Update local recent params
        prev_meta = self._params.get('LOCAL_RECENT')
        if prev_meta is None or time >= prev_meta.time:
            self._params['LOCAL_RECENT'] = param_meta

        # Update local best params
        prev_meta = self._params.get('LOCAL_BEST')
        if prev_meta is None or score >= prev_meta.score:
            self._params['LOCAL_BEST'] = param_meta

        # Update global recent params
        prev_meta = self._params.get('GLOBAL_RECENT')
        if prev_meta is None or time >= prev_meta.time:
            self._params['GLOBAL_RECENT'] = param_meta

        # Update global best params
        prev_meta = self._params.get('GLOBAL_BEST')
        if prev_meta is None or score >= prev_meta.score:
            self._params['GLOBAL_BEST'] = param_meta

        return param_meta

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
        param_meta = self._params.get('LOCAL_RECENT')
        if param_meta is None:
            return None

        return param_meta.param_id

    def _get_local_best_params(self):
        param_meta = self._params.get('LOCAL_BEST')
        if param_meta is None:
            return None
        
        return param_meta.param_id

    def _get_global_recent_params(self):
        param_meta = self._params.get('GLOBAL_RECENT')
        if param_meta is None:
            return None

        return param_meta.param_id

    def _get_global_best_params(self):
        param_meta = self._params.get('GLOBAL_BEST')
        if param_meta is None:
            return None

        return param_meta.param_id

    ####################################
    # Redis communication
    ####################################

    # Pulls metadata from Redis, updating local metadata
    def _pull_from_redis(self):
        redis_params = self._pull_metadata_from_redis()

        # Merge with local params meta
        for (param_type, param_meta) in redis_params.items():
            self._params[param_type] = param_meta
        
    # Pushes metadata & selected params to Redis, deletes outdated params on Redis
    def _push_to_redis(self):        
        params_to_push = ['GLOBAL_BEST', 'GLOBAL_RECENT']

        # Extract params meta to share
        params_shared = { param_type: param_meta for (param_type, param_meta) in self._params.items() if param_type in params_to_push }
        
        # Compare new against old params, and determine which params to push and delete from Redis 
        redis_params = self._pull_metadata_from_redis()
        og_param_ids = set([x.param_id for x in redis_params.values()])
        new_param_ids = set([x.param_id for x in params_shared.values()])
        to_add = [x for x in new_param_ids if x not in og_param_ids]
        to_delete = [x for x in og_param_ids if x not in new_param_ids]

        # For each param to add, push it
        for param_id in to_add:
            params = self._params_cache.get(param_id)
            if params:
                self._push_params_to_redis(param_id, params)
        
        # Delete params to delete
        self._delete_params_from_redis(*to_delete)

        # Push updated metadata to Redis
        self._push_metadata_to_redis(params_shared)
    
    def _push_metadata_to_redis(self, params):
        meta_name = self._get_redis_name('meta')
        redis_params = { param_type: self._param_meta_to_jsonable(param_meta) for (param_type, param_meta) in params.items() }
        metadata = {
            'params': redis_params
        }
        logger.info('Pushing metadata to Redis: {}...'.format(metadata))

        metadata_str = json.dumps(metadata)
        self._redis.set(meta_name, metadata_str)

    def _pull_metadata_from_redis(self):
        meta_name = self._get_redis_name('meta')
        metadata_str = self._get_from_redis(meta_name)

        # Pull metadata from redis
        if metadata_str is not None:
            metadata = json.loads(metadata_str)
            logger.info('Pulled metadata from Redis: {}'.format(metadata))

            # For each param stored on Redis, update its metadata
            params = metadata.get('params', {})
            params = { param_type: self._jsonable_to_param_meta(jsonable) for (param_type, jsonable) in params.items() }
            return params

        return {}

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

