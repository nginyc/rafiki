import os
import uuid
from importlib import import_module

from rafiki.constants import ModelDependency

def load_model_class(model_file_bytes, model_class):
    temp_mod_name = str(uuid.uuid4())
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
    commands = []

    # Determine PIP packages to install
    pip_packages = []
    for (dep, ver) in dependencies.items():
        if dep == ModelDependency.KERAS:
            pip_packages.append('Keras=={}'.format(ver))
        elif dep == ModelDependency.PYTORCH:
            pip_packages.append('torch=={}'.format(ver))
        elif dep == ModelDependency.SCIKIT_LEARN:
            pip_packages.append('scikit-learn=={}'.format(ver))
        elif dep == ModelDependency.TENSORFLOW:
            if enable_gpu:
                pip_packages.append('tensorflow-gpu=={}'.format(ver))
            else:
                pip_packages.append('tensorflow=={}'.format(ver))
    
    if len(pip_packages) > 0:
        commands.append('pip install {};'.format(' '.join(pip_packages)))

    return ' '.join(commands)



    