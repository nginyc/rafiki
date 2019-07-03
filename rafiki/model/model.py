import os
import json
import abc
import traceback
import pickle
import uuid
from importlib import import_module
import inspect

from rafiki.advisor import Advisor, AdvisorType
from rafiki.predictor import ensemble_predictions
from rafiki.constants import TaskType, ModelDependency

from .dataset import ModelDatasetUtils
from .knob import BaseKnob, serialize_knob_config, deserialize_knob_config

class InvalidModelClassException(Exception): pass
class InvalidModelParamsException(Exception): pass

class BaseModel(abc.ABC):
    '''
    Rafiki's base model class that Rafiki models should extend. 
    Rafiki models should implement all abstract methods according to their associated tasks' specifications,
    together with the static method ``get_knob_config()``.

    In the model's ``__init__`` method, call ``super().__init__(**knobs)`` as the first line, 
    followed by the model's initialization logic. The model should be initialize itself with ``knobs``, 
    a set of generated knob values for the instance, and possibly save the knobs' values as 
    attribute(s) of the model instance. These knob values will be chosen by Rafiki based on the model's knob config. 
    
    For example:

    ::

        def __init__(self, **knobs):
            super().__init__(**knobs)
            self.__dict__.update(knobs)
            ...
            self._build_model(self.knob1, self.knob2)


    :param knobs: Dictionary of knob values for this model instance
    :type knobs: dict[str, any]
    '''   
    def __init__(self, **knobs):
        pass

    @staticmethod
    def get_knob_config():
        '''
        Return a dictionary defining this model class' knob configuration 
        (i.e. list of knob names, their data types and their ranges).

        :returns: Dictionary defining this model's knob configuration 
        :rtype: dict[str, rafiki.model.BaseKnob]
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self, dataset_uri):
        '''
        Train this model instance with given dataset and initialized knob values.

        :param str dataset_uri: URI of the train dataset in a format specified by the task
        '''
        raise NotImplementedError()

    # TODO: Allow configuration of other metrics
    @abc.abstractmethod
    def evaluate(self, dataset_uri):
        '''
        Evaluate this model instance with given dataset after training. 
        This will be called only when model is *trained*.

        :param str dataset_uri: URI of the test dataset in a format specified by the task
        :returns: Accuracy as float from 0-1 on the test dataset
        :rtype: float
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, queries):
        '''
        Make predictions on a batch of queries with this model instance after training. 
        Each prediction should be JSON serializable.
        This will be called only when model is *trained*.

        :param queries: List of queries, where a query is in a format specified by the task 
        :type queries: list[any]
        :returns: List of predictions, where a prediction is in a format specified by the task 
        :rtype: list[any]
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def dump_parameters(self):
        '''
        Return a dictionary of model parameters that fully defines this model instance's trained state. 
        This dictionary should be serializable by the Python's ``pickle`` module.
        This will be used for trained model serialization within Rafiki.
        This will be called only when model is *trained*.

        :returns: Dictionary of model parameters
        :rtype: dict[string, any]
        '''

        raise NotImplementedError()

    @abc.abstractmethod
    def load_parameters(self, params):
        '''
        Load a dictionary of model parameters into this model instance.
        This will be used for trained model deserialization within Rafiki.
        The model will be considered *trained* subsequently.

        :param params: Dictionary of model parameters
        :type params: dict[string, any]
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy(self):
        '''
        Destroy this model instance, closing any sessions or freeing any connections.
        No other methods will be called subsequently.
        '''
        pass

def test_model_class(model_file_path, model_class, task, dependencies, \
                    train_dataset_uri, test_dataset_uri, \
                    queries=[], knobs=None, features=None, target=None):
    '''
    Tests whether a model class is properly defined by running a full train-inference flow.
    The model instance's methods will be called in an order similar to that in Rafiki.
    It is assumed that all of the model's dependencies have been installed in the current Python environment. 

    :param str model_file_path: Path to a single Python file that contains the definition for the model class
    :param type model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
    :param str task: Task type of model
    :param dependencies: Model's dependencies
    :type dependencies: dict[str, str]
    :param str train_dataset_uri: URI of the train dataset for testing the training of model
    :param str test_dataset_uri: URI of the test dataset for testing the evaluating of model
    :param list[any] queries: List of queries for testing predictions with the trained model
    :param knobs: Knobs to train the model with. If not specified, knobs from an advisor will be used
    :type knobs: dict[str, any]
    :param list[str] features: List of features for tabular dataset
    :param str target: target column to predict for tabular dataset
    :returns: The trained model
    '''
    try:
        _print_header('Checking model dependencies...')
        _check_dependencies(dependencies)

        _print_header('Checking loading of model & model definition...')
        f = open(model_file_path, 'rb')
        model_file_bytes = f.read()
        f.close()
        py_model_class = load_model_class(model_file_bytes, model_class, temp_mod_name='your-model-file-temp')
        _check_model_class(py_model_class)

        _print_header('Checking model knob configuration...')
        knob_config = py_model_class.get_knob_config()
        _check_knob_config(knob_config)

        _print_header('Checking model initialization...')
        advisor = Advisor(knob_config, advisor_type=AdvisorType.BTB_GP)
        if knobs is None: knobs = advisor.propose()
        print('Using knobs: {}'.format(knobs))
        model_inst = py_model_class(**knobs)
        _check_model_inst(model_inst)

        _print_header('Checking training & evaluation of model...')
        if task == TaskType.TABULAR_REGRESSION or TaskType.TABULAR_CLASSIFICATION:
            model_inst.train(train_dataset_uri, features, target)
            score = model_inst.evaluate(test_dataset_uri, features, target)
        else:
            model_inst.train(train_dataset_uri)
            score = model_inst.evaluate(test_dataset_uri)

        if not isinstance(score, float):
            raise Exception('`evaluate()` should return a float!')

        print('Score: {}'.format(score))

        _print_header('Checking dumping of parameters of model...')
        parameters = model_inst.dump_parameters()

        if not isinstance(parameters, dict):
            raise Exception('`dump_parameters()` should return a dict[str, any]')

        try:
            # Model parameters are pickled and put into DB
            parameters = pickle.loads(pickle.dumps(parameters))
        except Exception:
            traceback.print_stack()
            raise Exception('`parameters` should be serializable by `pickle`')

        _print_header('Checking loading of parameters of model...')
        model_inst.destroy()
        model_inst = py_model_class(**knobs)
        model_inst.load_parameters(parameters)

        if queries != []:
            _print_header('Checking predictions with model...')
            print('Using queries: {}'.format(queries))
            predictions = model_inst.predict(queries)

            try:
                for prediction in predictions:
                    json.dumps(prediction)
            except Exception:
                traceback.print_stack()
                raise Exception('Each `prediction` should be JSON serializable')

            # Ensembling predictions in predictor
            predictions = ensemble_predictions([predictions], task)

            print('Predictions: {}'.format(predictions))

        _note('The model definition is valid!')
    
        return model_inst

    except Exception as e:
        raise InvalidModelClassException(e)

def load_model_class(model_file_bytes, model_class, temp_mod_name=None):
    if temp_mod_name is None:
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
    conda_env = os.environ.get('CONDA_ENVIORNMENT')
    commands = []

    # Determine install commands for each dependency
    for (dep, ver) in dependencies.items():
        if dep == ModelDependency.KERAS:
            commands.append('pip install Keras=={}'.format(ver))
        elif dep == ModelDependency.PYTORCH:
            commands.append('pip install torch=={}'.format(ver))
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

def _check_dependencies(dependencies):

    if not isinstance(dependencies, dict):
        raise Exception('`dependencies` should be a dict[str, str]')

    for (dep, ver) in dependencies.items():
        if dep == ModelDependency.KERAS:
            # Warn that Keras models should additionally depend on TF for GPU usage
            _note('Keras models can enable GPU usage with by adding a `tensorflow` dependency.')
        elif dep == ModelDependency.PYTORCH:
            pass
        elif dep == ModelDependency.SCIKIT_LEARN:
            pass
        elif dep == ModelDependency.TENSORFLOW:
            _note('TensorFlow models must cater for GPU-sharing with ' \
                    + '`config.gpu_options.allow_growth = True` (ref: https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth).')
        elif dep == ModelDependency.SINGA:
            pass
    
    install_command = parse_model_install_command(dependencies, enable_gpu=False)
    install_command_with_gpu = parse_model_install_command(dependencies, enable_gpu=True)
    _note(f'Install command (without GPU): `{install_command}`')
    _note(f'Install command (with GPU): `{install_command_with_gpu}`')

def _check_model_class(py_model_class):
    if not issubclass(py_model_class, BaseModel):
        raise Exception('Model should extend `rafiki.model.BaseModel`')

    if inspect.isfunction(getattr(py_model_class, 'init', None)):
        _warn('`init` has been deprecated - use `__init__` for your model\'s initialization logic instead')

    if inspect.isfunction(getattr(py_model_class, 'get_knob_config', None)) and \
        not isinstance(py_model_class.__dict__.get('get_knob_config', None), staticmethod):
        _warn('`get_knob_config` has been changed to a `@staticmethod`')

def _check_model_inst(model_inst):
    # Throw error when deprecated methods are called
    def deprecated_func(desc):
        def throw_error(*args, **kwargs):
            raise AttributeError(desc)
        
        return throw_error

    class DeprecatedModelUtils():
        log = deprecated_func('`self.utils.log(...)` has been moved to `logger.log(...)`')
        log_metrics = deprecated_func('`self.utils.log_metrics(...)` has been moved to `logger.log(...)`')
        define_plot = deprecated_func('`self.utils.define_plot(...)` has been moved to `logger.define_plot(...)`')
        define_loss_plot = deprecated_func('`self.utils.define_loss_plot(...)` has been moved to `logger.define_loss_plot(...)`')
        log_loss_metric = deprecated_func('`self.utils.log_loss_metric(...)` has been moved to `logger.log_loss(...)`')
        load_dataset_of_image_files = deprecated_func('`self.utils.load_dataset_of_image_files(...)` has been moved to `dataset_utils.load_dataset_of_image_files(...)`')
        load_dataset_of_corpus = deprecated_func('`self.utils.load_dataset_of_corpus(...)` has been moved to `dataset_utils.load_dataset_of_corpus(...)`')
        resize_as_images = deprecated_func('`self.utils.resize_as_images(...)` has been moved to `dataset_utils.resize_as_images(...)`')
        download_dataset_from_uri = deprecated_func('`self.utils.download_dataset_from_uri(...)` has been moved to `dataset_utils.download_dataset_from_uri(...)`')

    model_inst.utils = DeprecatedModelUtils()

def _check_knob_config(knob_config):
    if not isinstance(knob_config, dict) or \
        any([(not isinstance(name, str) or not isinstance(knob, BaseKnob)) for (name, knob) in knob_config.items()]):
        raise Exception('Static method `get_knob_config()` should return a dict[str, BaseKnob]')

    # Try serializing and deserialize knob config
    knob_config_str = serialize_knob_config(knob_config)
    knob_config = deserialize_knob_config(knob_config_str)

def _print_header(msg):
    print('-' * (len(msg) + 4))
    print('| {} |'.format(msg))
    print('-' * (len(msg) + 4))

def _warn(msg):
    print(f'\033[93mWARNING: {msg}\033[0m')

def _note(msg):
    print(f'\033[94mNOTE: {msg}\033[0m')