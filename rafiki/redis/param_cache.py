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

import json
import numpy as np
from typing import Dict
import uuid
import logging
from collections import namedtuple
import msgpack
from datetime import datetime
import traceback

from rafiki.model import Params
from rafiki.advisor import ParamsType
from rafiki.utils.local_cache import LocalCache
from .redis import RedisSession

logger = logging.getLogger(__name__)

class InvalidParamsError(Exception): pass
class InvalidParamsFormatError(Exception): pass

REDIS_NAMESPACE = 'PARAMS'
PARAM_DATA_TYPE_SEPARATOR = '//'
PARAM_DATA_TYPE_NUMPY = 'NP'

_ParamMeta = namedtuple('_ParamMeta', ('param_id', 'score', 'time'))

class ParamCache(object):
    '''
    Retrieves and caches parameters for a session & a worker, backed by an in-memory cache and Redis for cross-worker sharing (optional).

    :param str session_id: Session ID associated with the parameters
    '''

    '''
        Internally, organises data into these namespaces:
            
        params:<param_id>   | Params by ID
        meta                | Aggregate of all global metadata: 
                            { params:   { GLOBAL_BEST:     { score, time, param_id }, 
                                        { GLOBAL_RECENT:   { score, time, param_id }   }            
    '''
    
    def __init__(self, 
                session_id='local', 
                redis_host=None,
                redis_port=None, 
                cache_size=4):
        self._params: Dict[str, _ParamMeta] = {} # Stores params metadata
        redis_namespace = f'{REDIS_NAMESPACE}:{session_id}'
        self._redis = RedisSession(redis_namespace, redis_host, redis_port)
        self._local_cache = LocalCache(cache_size)

    '''
    Stores parameters into underlying storage.

    :param Params params: Parameters as a { <name>: <numpy array> } dictionary
    :param datetime time: When the parameters were produced
    :param float score: Score associated with the parameters
    '''
    def store_params(self, params: Params, score: float = None, time: datetime = None):
        if params is None:
            raise InvalidParamsError('`params` cannot be `None`')   

        self._redis.acquire_lock() 
        try:
            # With redis, sync in-memory metadata with Redis'
            self._pull_from_redis()

            param_meta = self._update_params_meta(score, time)
            if param_meta:
                # Store input params in in-memory cache
                self._local_cache.put(param_meta.param_id, params)

            if self._redis:
                self._push_to_redis()

        finally:
            self._redis.release_lock() 

    '''
    Retrieves parameters from underlying storage.

    :param ParamsType params_type: Type of parameters to retrieve
    :returns: Parameters as a { <name>: <numpy array> } dictionary
    :rtype: Params
    '''
    def retrieve_params(self, params_type: ParamsType) -> Params:
        self._redis.acquire_lock() 
        try:
            # With redis, sync in-memory metadata with Redis'
            self._pull_from_redis()

            # Get param id to fetch
            param_id = self._get_params_by_type(params_type)
            if param_id is None:
                return None

            logger.info('To use params "{}"'.format(param_id))

            # Check in cache first
            params = self._local_cache.get(param_id)
            if params is not None:
                return params

            # Check in redis next, and store it in cache
            params = self._pull_params_from_redis(param_id)
            if params is None:
                logger.error('Params don\'t exist in Redis!')
                return None

            self._local_cache.put(param_id, params)
            return params

        finally:
            self._redis.release_lock() 

    '''
    Clears all parameters for this session from underlying storage.
    '''
    def clear_all_params(self):
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
            params = self._local_cache.get(param_id)
            if params:
                self._push_params_to_redis(param_id, params)
        
        # Delete params to delete
        if len(to_delete) > 0:
            self._delete_params_from_redis(*to_delete)

        # Push updated metadata to Redis
        self._push_metadata_to_redis(params_shared)
    
    def _push_metadata_to_redis(self, params):
        redis_params = { param_type: self._param_meta_to_jsonable(param_meta) for (param_type, param_meta) in params.items() }
        metadata = {
            'params': redis_params
        }
        logger.info('Pushing metadata to Redis: {}...'.format(metadata))

        metadata_str = json.dumps(metadata)
        self._redis.set('meta', metadata_str)

    def _pull_metadata_from_redis(self):
        metadata_str = self._redis.get('meta')

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
        param_names = ['params:{}'.format(x) for x in param_ids]
        self._redis.delete(*param_names)

    # Clears ALL metadata and params for session from Redis
    def _clear_all_from_redis(self):       
        logger.info('Clearing metadata and params from Redis...')
        self._redis.delete('meta')
        self._redis.delete_pattern('params:*')

    def _push_params_to_redis(self, param_id: str, params: Params):
        logger.info('Pushing params: "{}"...'.format(param_id))
        param_name = 'params:{}'.format(param_id)
        params_bytes = _serialize_params(params)
        self._redis.set(param_name, params_bytes)

    def _pull_params_from_redis(self, param_id: str) -> Params:
        logger.info('Pulling params: "{}"...'.format(param_id))
        param_name = 'params:{}'.format(param_id)
        params_bytes = self._redis.get(param_name)
        if params_bytes is None:
            return None
            
        params = _deserialize_params(params_bytes)
        return params

    def _param_meta_to_jsonable(self, param_meta: _ParamMeta):
        jsonable = param_meta._asdict()
        jsonable['time'] = str(jsonable['time'])
        return jsonable

    def _jsonable_to_param_meta(self, jsonable):
        jsonable['time'] = datetime.strptime(jsonable['time'], '%Y-%m-%d %H:%M:%S.%f')
        param_meta = _ParamMeta(**jsonable)
        return param_meta

 
def _serialize_params(params):
    # Serialize as `msgpack`
    params_simple = _simplify_params(params)
    params_bytes = msgpack.packb(params_simple, use_bin_type=True)
    return params_bytes

def _deserialize_params(params_bytes):
    # Deserialize as `msgpack`
    params_simple = msgpack.unpackb(params_bytes, raw=False)
    params = _unsimplify_params(params_simple)
    return params

def _simplify_params(params):
    try:
        params_simple = {}

        assert isinstance(params, dict)
        for (name, value) in params.items():
            assert isinstance(name, str)
            assert PARAM_DATA_TYPE_SEPARATOR not in name # Internally used as separator for types

            # If value is a numpy array, prefix it with type
            # Otherwise, it must be one of the basic types
            if isinstance(value, np.ndarray):
                name = f'{PARAM_DATA_TYPE_NUMPY}{PARAM_DATA_TYPE_SEPARATOR}{name}'
                value = value.tolist()
            else:
                assert isinstance(value, (str, float, int))

            params_simple[name] = value

        return params_simple

    except:
        traceback.print_stack()
        raise InvalidParamsFormatError()

def _unsimplify_params(params_simple):
    params = {}

    for (name, value) in params_simple.items():
        if PARAM_DATA_TYPE_SEPARATOR in name:
            (type_id, name) = name.split(PARAM_DATA_TYPE_SEPARATOR)
            if type_id == PARAM_DATA_TYPE_NUMPY:
                value = np.array(value)

        params[name] = value

    return params