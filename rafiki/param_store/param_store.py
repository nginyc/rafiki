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

import abc
import msgpack
import traceback
import numpy as np

from rafiki.model import Params

class InvalidParamsFormatError(Exception): pass

PARAM_DATA_TYPE_SEPARATOR = '//'
PARAM_DATA_TYPE_NUMPY = 'NP'

class ParamStore(abc.ABC):
    '''
        Persistent store for model parameters.
    '''

    @abc.abstractmethod
    def save(self, params: Params) -> str:
        '''
            Persists a set of parameters, returning a unique ID for the params.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, params_id: str) -> Params:
        '''
            Loads persisted parameters, identified by ID.
        '''
        raise NotImplementedError()

    @staticmethod
    def _serialize_params(params):
        # Serialize as `msgpack`
        params_simple = _simplify_params(params)
        params_bytes = msgpack.packb(params_simple, use_bin_type=True)
        return params_bytes

    @staticmethod
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