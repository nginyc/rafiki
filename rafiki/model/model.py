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
from .log import ModelLogUtils
from .knob import BaseKnob

class InvalidModelClassException(Exception): pass
class InvalidModelParamsException(Exception): pass

class ModelUtils(ModelDatasetUtils, ModelLogUtils):
    def __init__(self):
        ModelDatasetUtils.__init__(self)
        ModelLogUtils.__init__(self)

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
        self.utils = ModelUtils()

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
        Return a dictionary of model parameters that fully define this model instance's trained state. 
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
                    enable_gpu=False, queries=[], knobs=None):
    '''
    Tests whether a model class is properly defined by running a full train-inference flow.
    The model instance's methods will be called in an order similar to that in Rafiki.

    :param str model_file_path: Path to a single Python file that contains the definition for the model class
    :param obj model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
    :param str task: Task type of model
    :param dict[str, str] dependencies: Model's dependencies
    :param str train_dataset_uri: URI of the train dataset for testing the training of model
    :param str test_dataset_uri: URI of the test dataset for testing the evaluating of model
    :param list[any] queries: List of queries for testing predictions with the trained model
    :param knobs: Knobs to train the model with. If not specified, knobs from an advisor will be used
    :type knobs: dict[str, any]
    :returns: The trained model
    '''
    try:
        _print_header('Installing & checking model dependencies...')
        _check_dependencies(dependencies)

        # Test installation
        if not isinstance(dependencies, dict):
            raise Exception('`dependencies` should be a dict[str, str]')

        install_command = parse_model_install_command(dependencies, enable_gpu=enable_gpu)
        exit_code = os.system(install_command)
        if exit_code != 0: raise Exception('Error in installing model dependencies')

        _print_header('Checking loading of model & model definition...')
        f = open(model_file_path, 'rb')
        model_file_bytes = f.read()
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

        _info('The model definition is valid!')
    
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

def _check_dependencies(dependencies):
    for (dep, ver) in dependencies.items():
        # Warn that TF models need to cater for GPU sharing
        if dep == ModelDependency.TENSORFLOW:
            _info('`tensorflow-gpu` of the same version will be installed if GPU is available during training.')
            _warn('TensorFlow models must cater for GPU-sharing with ' \
                    + '`config.gpu_options.allow_growth = True` (ref: https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth).')

        # Warn that Keras models should additionally depend on TF for GPU usage
        elif dep == ModelDependency.KERAS:
            _warn('Keras models can enable GPU usage with by adding a `tensorflow` dependency.')

def _check_model_class(py_model_class):
    if not issubclass(py_model_class, BaseModel):
        raise Exception('Model should extend `rafiki.model.BaseModel`')

    if inspect.isfunction(getattr(py_model_class, 'get_predict_label_mapping', None)):
        _warn('`get_predict_label_mapping` has been deprecated')
    
    if inspect.isfunction(getattr(py_model_class, 'init', None)):
        _warn('`init` has been deprecated - use `__init__` for your model\'s initialization logic instead')

    if inspect.isfunction(getattr(py_model_class, 'get_knob_config', None)) and \
        not isinstance(py_model_class.__dict__.get('get_knob_config', None), staticmethod):
        _warn('`get_knob_config` has been changed to a `@staticmethod`')

def _check_model_inst(model_inst):
    if getattr(model_inst, 'utils', None) is None:
        raise Exception('`super().__init__(**knobs)` should be called as the first line of the model\'s `__init__` method.')

def _check_knob_config(knob_config):
    if not isinstance(knob_config, dict) or \
        any([(not isinstance(name, str) or not isinstance(knob, BaseKnob)) for (name, knob) in knob_config.items()]):
        raise Exception('Static method `get_knob_config()` should return a dict[str, BaseKnob]')

def _info(msg):
    msg_color = '\033[94m'
    end_color = '\033[0m'
    print('{}{}{}'.format(msg_color, msg, end_color))

def _print_header(msg):
    print('-' * (len(msg) + 4))
    print('| {} |'.format(msg))
    print('-' * (len(msg) + 4))

def _warn(msg):
    msg_color = '\033[93m'
    end_color = '\033[0m'
    print('{}WARNING: {}{}'.format(msg_color, msg, end_color))