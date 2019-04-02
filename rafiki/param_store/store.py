import os
import json
import numpy as np
from typing import Dict
import uuid
import logging

from rafiki.model import Params

from .cache import Cache

logger = logging.getLogger(__name__)

class InvalidParamsError(Exception): pass

class ParamStore(object):
    REDIS_NAMESPACE = 'PARAMS'

    '''
    Store API that retrieves and stores parameters, backed an in-memory cache and Redis (optional).
    '''
    def __init__(self, cache_size=128, redis_host=None, redis_port=6379):
        self._redis = self._make_redis_client(redis_host, redis_port) if redis_host is not None else None
        self._cache = Cache(cache_size)
    
    '''
    Retrieves parameters for a session from underlying storage.

    :param str session_id: Unique session ID for parameters
    :param str param_id: ID for parameters
    :returns: Parameters as a { <name>: <numpy array> } dictionary
    :rtype: Params
    '''
    def retrieve_params(self, session_id: str, param_id: str) -> Params:
        # Check in cache first
        params = self._cache.get(param_id)
        if params is not None:
            return params

        if self._redis is None:
            return {}

        # Check in redis next, fetching the whole params dict associated with the param
        logger.info('Fetching params "{}" from Redis...'.format(param_id))
        fetched_params_str = self._redis.get(param_id)
        if fetched_params_str is None:
            logger.info('Params don\'t exist in Redis')
            return {}
        
        # Store fetched params in redis
        fetched_params = self._deserialize_params(fetched_params_str)
        self._cache.put(param_id, fetched_params)

        # Check cache again
        params = self._cache.get(param_id)
        if params is not None:
            return params
        
        return {}

    '''
    Stores parameters for a session into underlying storage.

    :param str session_id: Unique session ID for parameters
    :param Params params: Parameters as a { <name>: <numpy array> } dictionary
    :param str trial_id: Associated trial ID for parameters
    :returns: ID for parameters
    :rtype: str
    '''
    def store_params(self, session_id: str, params: Params, trial_id: str = None) -> str:
        if params is None:
            raise InvalidParamsError('`params` cannot be `None`')    

        trial_id = trial_id or uuid.uuid4()
        session_key = '{}:{}'.format(self.REDIS_NAMESPACE, session_id) 
        param_id = '{}:{}'.format(session_key, trial_id) # <namespace>:<session_id>:<trial_id>
        
        if self._redis is not None:
            # Store params dict in redis
            params_str = self._serialize_params(params)
            logger.info('Storing params "{}" into Redis...'.format(param_id))
            self._redis.set(param_id, params_str)

        self._cache.put(param_id, params)
        return param_id

    '''
    Clears all parameters for a session from underlying storage.

    :param str session_id: Unique session ID for parameters
    '''
    def clear_params(self, session_id: str):
        session_key = '{}:{}'.format(self.REDIS_NAMESPACE, session_id) 

        # Clear params from redis
        if self._redis is not None:
            params_keys = self._redis.keys('{}:*'.format(session_key))
            if len(params_keys) > 0:
                logger.info('Clearing {} params for session "{}" from Redis...'.format(len(params_keys), session_id))
                self._redis.delete(*params_keys)

    def _make_redis_client(self, host, port):
        import redis
        cache_connection_url = 'redis://{}:{}'.format(host, port)
        connection_pool = redis.ConnectionPool.from_url(cache_connection_url)
        client = redis.StrictRedis(connection_pool=connection_pool, decode_responses=True)
        return client

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

        # Convert lists to numpy arrays
        for (name, value) in params.items():
            if isinstance(value, list):
                params[name] = np.asarray(value)
        
        return params

