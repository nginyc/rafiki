import os
import uuid
from typing import Type
from importlib import import_module
import pickle

from rafiki.constants import ModelDependency

from .model import BaseModel
from .dataset import DatasetUtils
from .log import LoggerUtils

def load_model_class(model_file_bytes, model_class, temp_mod_name=None) -> Type[BaseModel]:
    if temp_mod_name is None:
        temp_mod_name = '{}-{}'.format(model_class, str(uuid.uuid4()))

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
        raise e
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
    
