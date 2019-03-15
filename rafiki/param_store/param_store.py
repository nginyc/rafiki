import redis
import os
import json
import numpy as np
from typing import Dict
import uuid
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class ParamStore(object):
    REDIS_NAMESPACE = 'PARAMS'

    '''
    Store API that retrieves and stores parameters, backed an in-memory cache and Redis (optional).
    '''
    def __init__(self, cache_size=16384, redis_host=None, redis_port=6379):
        self._redis = self._make_redis_client(redis_host, redis_port) if redis_host is not None else None
        self._cache = _ParamCache(cache_size)
    
    '''
    Retrieves parameters for a session from underlying storage.

    :param str session_id: Unique session ID for parameters
    :param dict params: Parameters as a light { <name>: <id> } dictionary
    :returns: Parameters as a heavy { <name>: <numpy array> } dictionary
    :rtype: dict
    '''
    def retrieve_params(self, session_id: str, params: Dict[str, str]):
        out_params = {}

        # For each param
        for (name, param_id) in params.items():

            # Check in cache first
            params = self._cache.get(param_id)
            if params is not None:
                out_params[name] = params
                continue
            
            # Check in redis next, fetching the whole params dict associated with the param
            params_key = ':'.join(param_id.split(':')[0:3]) # <namespace>:<session_id>:<prefix>
            logger.info('Fetching key "{}" for session "{}" from Redis...'.format(params_key, session_id))
            params_str = self._redis.get(params_key)
            if params_str is None:
                logger.info('Key doesn\'t exist in Redis')
                continue

            # Store the whole params dict in cache
            params = self._deserialize_params(params_str)
            for (name, value) in params.items():
                fetched_param_id = '{}:{}'.format(params_key, name)
                self._cache.put(fetched_param_id, value)

            # Check cache again
            params = self._cache.get(param_id)
            if params is not None:
                out_params[name] = params
                
        return out_params

    '''
    Stores parameters for a session into underlying storage.

    :param str session_id: Unique session ID for parameters
    :param dict params: Parameters as a heavy { <name>: <numpy array> } dictionary
    :param str prefix: Prefix for each parameter's name to make parameter names unique across different calls
    :returns: Parameters as a light { <name>: <id> } dictionary
    :rtype: dict
    '''
    def store_params(self, session_id: str, params: Dict[str, np.array], prefix: str = None):
        prefix = prefix or uuid.uuid4()
        session_key = '{}:{}'.format(self.REDIS_NAMESPACE, session_id) 
        params_key = '{}:{}'.format(session_key, prefix) # <namespace>:<session_id>:<prefix>
        
        if self._redis is not None:
            # Store whole params dict in redis
            params_str = self._serialize_params(params)
            logger.info('Storing key "{}" for session "{}" into Redis...'.format(params_key, session_id))
            self._redis.set(params_key, params_str)

        # Store param one by one in cache
        out_params = {}
        for (name, value) in params.items():
            param_id = '{}:{}'.format(params_key, name) # <namespace>:<session_id>:<prefix>:<param_name>
            self._cache.put(param_id, value)
            out_params[name] = param_id

        return out_params

    '''
    Clears all parameters for a session from underlying storage.

    :param str session_id: Unique session ID for parameters
    '''
    def clear_params(self, session_id: str):
        session_key = '{}:{}'.format(self.REDIS_NAMESPACE, session_id) 

        # Clear params from redis
        if self._redis is not None:
            params_keys = self._redis.keys('{}:*'.format(session_key))
            logger.info('Clearing {} keys for session "{}" from Redis...'.format(len(params_keys), session_id))
            self._redis.delete(*params_keys)

    def _make_redis_client(self, host, port):
        cache_connection_url = 'redis://{}:{}'.format(host, port)
        connection_pool = redis.ConnectionPool.from_url(cache_connection_url)
        client = redis.StrictRedis(connection_pool=connection_pool, decode_responses=True)
        return client

    def _serialize_params(self, params):
        # Convert numpy arrays to lists
        for (name, value) in params.items():
            params[name] = list(value)

        # Convert to JSON
        params_str = json.dumps(params)
        return params_str

    def _deserialize_params(self, params_str):
        # Convert from JSON
        params = json.loads(params_str)

        # Convert lists to numpy arrays
        for (name, value) in params.items():
            params[name] = np.array(value)
        
        return params

class _ParamCache():
    def __init__(self, size: int):
        self._temp_result = None # For temporarily piping result into cache
        def get_from_cache(key):
            return self._temp_result
        self._get_from_cache = lru_cache(maxsize=size)(get_from_cache)

    def put(self, key: str, value):
        self._temp_result = value
        self._get_from_cache(key)
        self._temp_result = None

    def get(self, key: str):
        return self._get_from_cache(key)

