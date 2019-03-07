import os
import numpy as np
from typing import Dict
from functools import lru_cache

class ParamStore(object):
    '''
    Store API that retrieves and stores parameters.
    '''
    CACHE_SIZE = 16384 # Maximum no. of parameters to store in-memory

    def __init__(self):
        # Stores parameters in-memory
        self._cache = Cache(self.CACHE_SIZE)
    
    '''
    Retrieves parameters for a session from underlying storage.

    :param str session_id: Unique session ID for parameters
    :param dict params: Parameters as a light { <name>: <id> } dictionary
    :returns: Parameters as a heavy { <name>: <numpy array> } dictionary
    :rtype: dict
    '''
    def retrieve_params(self, session_id: str, params: Dict[str, str]):
        out_params = {}
        for (name, param_id) in params.items():
            key = '{}_{}'.format(session_id, param_id)
            params = self._cache.get(key)
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
    def store_params(self, session_id: str, params: Dict[str, np.array], prefix: str = ''):
        out_params = {}
        for (name, value) in params.items():
            param_id = '{}_{}'.format(prefix, name)
            key = '{}_{}'.format(session_id, param_id)
            self._cache.put(key, value)
            out_params[name] = param_id

        return out_params

class Cache():
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

