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

import os
import uuid
from typing import Type
from importlib import import_module
import pickle

from rafiki.constants import ModelDependency

from .model import BaseModel
from .dataset import DatasetUtils
from .log import LoggerUtils

class InvalidModelClassError(Exception): pass

def load_model_class(model_file_bytes, model_class, temp_mod_name=None) -> Type[BaseModel]:
    temp_mod_name = temp_mod_name or '{}-{}'.format(model_class, str(uuid.uuid4()))
    temp_model_file_name ='{}.py'.format(temp_mod_name)

    # Temporarily save the model file to disk
    with open(temp_model_file_name, 'wb') as f:
        f.write(model_file_bytes)

    try:
        # Import model file as module
        mod = import_module(temp_mod_name)
        # Extract model class from module
        clazz = getattr(mod, model_class)
    except Exception as e:
        raise InvalidModelClassError(e)
    finally:
        # Ensure that temp model file is removed upon model loading error
        os.remove(temp_model_file_name)

    return clazz

def parse_model_install_command(dependencies, enable_gpu=False):
    conda_env = os.environ.get('CONDA_ENVIORNMENT')
    commands = []

    # Determine install commands for each dependency
    for (dep, ver) in dependencies.items():
        if dep == ModelDependency.KERAS:
            commands.append('pip install Keras=={}'.format(ver))
        elif dep == ModelDependency.TORCH:
            commands.append('pip install torch=={}'.format(ver))
        elif dep == ModelDependency.TORCHVISION:
            commands.append('pip install torchvision=={}'.format(ver))
        elif dep == ModelDependency.SCIKIT_LEARN:
            commands.append('pip install scikit-learn=={}'.format(ver))
        elif dep == ModelDependency.XGBOOST:
            commands.append('pip install xgboost=={}'.format(ver))
        elif dep == ModelDependency.TENSORFLOW:
            if enable_gpu:
                commands.append('pip install tensorflow-gpu=={}'.format(ver))
            else:
                commands.append('pip install tensorflow=={}'.format(ver))
        elif dep == ModelDependency.SINGA:
            options = '-y -c nusdbsystem'
            if conda_env is not None:
                options += ' -n {}'.format(conda_env)
            if enable_gpu:
                commands.append('conda install {} singa-gpu={}'.format(options, ver))
            else:
                commands.append('conda install {} singa-cpu={}'.format(options, ver))
        else:
            # Assume that dependency is the exact PIP package name
            commands.append('pip install {}=={}'.format(dep, ver))

    return '; '.join(commands)

def deserialize_knob_config(knob_config_bytes):
    knob_config = pickle.loads(knob_config_bytes.encode())
    return knob_config

def serialize_knob_config(knob_config):
    knob_config_bytes = pickle.dumps(knob_config, 0).decode()
    return knob_config_bytes

class ModelUtils():
    def __init__(self):
        self._trial_id = None
        self.dataset = DatasetUtils()
        self.logger = LoggerUtils()

# Initialize a global instance
utils = ModelUtils()
logger = utils.logger
dataset = utils.dataset
    
