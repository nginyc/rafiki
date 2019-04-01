import os
import json
import abc
import traceback
import uuid
import inspect
import argparse
import time
import numpy as np
from collections import namedtuple
from datetime import datetime
from typing import Union, Dict, Type

from rafiki.model import BaseModel, BaseKnob, serialize_knob_config, deserialize_knob_config, \
                        parse_model_install_command, load_model_class, SharedParams, AvailableGpu
from rafiki.constants import TaskType, ModelDependency
from rafiki.param_store import ParamStore
from rafiki.predictor import ensemble_predictions

from .advisor import Advisor

class InvalidModelClassException(Exception): pass

def tune_model(py_model_class: Type[BaseModel], train_dataset_uri: str, val_dataset_uri: str, 
                test_dataset_uri: str = None, total_trials: int = 25, params_root_dir: str = 'params/', 
                advisor: Advisor = None, to_read_args: bool = True) -> (Dict[str, any], float, str):
    '''
    Tunes a model on a given dataset in the current environment.

    :param BaseModel py_model_class: The Python class for the model
    :param str train_dataset_uri: URI of the train dataset for testing the training of model
    :param str val_dataset_uri: URI of the validation dataset for evaluating a trained model
    :param str test_dataset_uri: URI of the validation dataset for testing the final best trained model, if provided
    :param int total_trials: Total number of trials to tune the model over
    :param str params_root_dir: Root folder path to create subfolders to save each trial's model parameters
    :param Advisor advisor: A pre-created advisor to use for tuning the model
    :param bool to_read_args: Whether should system args be read to retrieve default values for `total_trials` and knobs
    :rtype: (dict, float, str)
    :returns: (<knobs for best model>, <test score for best model>, <params directory for best model>)
    '''
    # Note start time
    start_time = time.time()

    # Retrieve config of model
    _print_header('Checking model configuration...')
    knob_config = py_model_class.get_knob_config()
    _check_knob_config(knob_config)

    # Maybe read from args
    if to_read_args:
        parser = argparse.ArgumentParser()
        parser.add_argument('--total_trials', type=int)
        (namespace_args, left_args) = parser.parse_known_args()
        total_trials = namespace_args.total_trials if namespace_args.total_trials is not None else total_trials  
        knobs_from_args = _maybe_read_knobs_from_args(knob_config, left_args)

    _info('Total trial count: {}'.format(total_trials))

    # Configure advisor
    if advisor is None:
        advisor = Advisor(knob_config)

    # Configure shared params monitor & store
    param_store = ParamStore()
    params_monitor = SharedParamsMonitor()
    
    # Variables to track over trials
    best_model_score = 0
    best_model_test_score = None
    best_model_knobs = None
    best_model_params_dir = None
    session_id = str(uuid.uuid4()) # Session ID for params store

    # Setup model class
    print('Running model class setup...')
    py_model_class.setup()

    # For every trial
    for i in range(1, total_trials + 1):
        trial_id = str(uuid.uuid4())
        _print_header('Trial #{} (ID: "{}")'.format(i, trial_id))

        # Get trial config
        trial_config = py_model_class.get_trial_config(i, total_trials, [])
        assert trial_config.is_valid

        # Get knobs proposal from advisor, overriding knobs from args & trial config
        knobs = advisor.propose()
        knobs = { **knobs, **trial_config.override_knobs, **knobs_from_args } 
        print('Advisor proposed knobs:', knobs)

        # Retrieve shared params from store
        param_id = params_monitor.get_params(trial_config.shared_params)
        params = {}
        if param_id is not None:
            print('To use {} shared params'.format(trial_config.shared_params.name))
            print('Retrieving shared params of ID "{}"...'.format(param_id))
            params = param_store.retrieve_params(session_id, param_id)

        # Load model
        model_inst = py_model_class(**knobs)

        # Train model
        print('Training model...')
        if len(params) > 0:
            model_inst.set_shared_parameters(params)
        model_inst.train(train_dataset_uri)
        trial_params = model_inst.get_shared_parameters() or None
        if trial_params:
            print('Model produced {} shared params'.format(len(trial_params)))

        # Evaluate model
        score = None
        if trial_config.should_evaluate:
            print('Evaluating model...')
            score = model_inst.evaluate(val_dataset_uri)
            if not isinstance(score, float):
                raise InvalidModelClassException('`evaluate()` should return a float!')

            print('Score on validation dataset:', score)

        # If trial has score
        if score is not None:
            # Update best model
            if score > best_model_score:
                _info('Best model so far! Beats previous best of score {}!'.format(best_model_score))
                       
                # Save best model
                params_dir = None
                if trial_config.should_save:
                    print('Saving trained model...')
                    params_dir = os.path.join(params_root_dir, trial_id + '/')
                    if not os.path.exists(params_dir):
                        os.mkdir(params_dir)
                    model_inst.save_parameters(params_dir)
                    _info('Model saved to {}'.format(params_dir))

                best_model_params_dir = params_dir
                best_model_knobs = knobs
                best_model_score = score
                        
                # Test best model, if test dataset provided
                if test_dataset_uri is not None:
                    print('Evaluting model on test dataset...')
                    best_model_test_score = model_inst.evaluate(test_dataset_uri)
                    _info('Score on test dataset: {}'.format(best_model_test_score))
                 
            
            # Feedback to advisor 
            advisor.feedback(score, knobs)

        # Update params monitor & store
        if trial_params:
            print('Storing shared params...')
            trial_param_id = param_store.store_params(session_id, trial_params, trial_id)
            params_monitor.add_params(trial_param_id, score)
            print('Stored shared params of ID "{}"'.format(trial_param_id))
    
    # Declare best model
    _info('Best model has knobs {} with score of {}'.format(best_model_knobs, best_model_score))
    if best_model_test_score is not None:
        _info('...with test score of {}'.format(best_model_test_score))
    if best_model_params_dir is not None:
        _info('...saved at {}'.format(best_model_params_dir)) 
        
    # Teardown model class
    print('Running model class teardown...')
    py_model_class.teardown()

    # Print duration
    duration = time.time() - start_time
    print('Tuning took a total of {}s'.format(duration))

    return (best_model_knobs, best_model_test_score, best_model_params_dir)

def test_model_class(model_file_path: str, model_class: str, task: str, dependencies: Dict[str, str],
                    train_dataset_uri: str, val_dataset_uri: str, enable_gpu: bool = False, queries: list = []):
    '''
    Tests whether a model class is properly defined by running a full train-inference flow.
    The model instance's methods will be called in an order similar to that in Rafiki.

    :param str model_file_path: Path to a single Python file that contains the definition for the model class
    :param str model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
    :param str task: Task type of model
    :param dict[str, str] dependencies: Model's dependencies
    :param str train_dataset_uri: URI of the train dataset for testing the training of model
    :param str val_dataset_uri: URI of the validation dataset for testing the evaluation of model
    :param bool enable_gpu: Whether to enable GPU during model training
    :param list[any] queries: List of queries for testing predictions with the trained model
    :type knobs: dict[str, any]
    :returns: The trained model
    '''
    _print_header('Installing & checking model dependencies...')
    _check_dependencies(dependencies)

    # Test installation
    if not isinstance(dependencies, dict):
        raise InvalidModelClassException('`dependencies` should be a dict[str, str]')

    install_command = parse_model_install_command(dependencies, enable_gpu=enable_gpu)
    exit_code = os.system(install_command)
    if exit_code != 0: 
        raise InvalidModelClassException('Error in installing model dependencies')

    _print_header('Checking loading of model & model definition...')
    with open(model_file_path, 'rb') as f:
        model_file_bytes = f.read()
    py_model_class = load_model_class(model_file_bytes, model_class, temp_mod_name=model_class)
    _check_model_class(py_model_class)

    (best_knobs, best_model_params_dir) = tune_model(py_model_class, train_dataset_uri, val_dataset_uri, total_trials=2)

    _print_header('Checking loading of parameters of model...')
    model_inst = py_model_class(**best_knobs)
    model_inst.load_parameters(best_model_params_dir)

    _print_header('Checking predictions with model...')
    print('Using queries: {}'.format(queries))
    predictions = model_inst.predict(queries)

    try:
        for prediction in predictions:
            json.dumps(prediction)
    except Exception:
        traceback.print_stack()
        raise InvalidModelClassException('Each `prediction` should be JSON serializable')

    # Ensembling predictions in predictor
    predictions = ensemble_predictions([predictions], task)

    print('Predictions: {}'.format(predictions))

    _info('The model definition is valid!')

    return model_inst

_Params = namedtuple('_Param', ('param_id', 'score', 'time'))

class SharedParamsMonitor():
    '''
    Monitors params across trials and maps a shared params policy to the exact params to use for trial
    '''
    _worker_to_best_params: Dict[str, _Params] = {}
    _worker_to_recent_params: Dict[str, _Params] = {}

    def add_params(self, param_id: str, score: float = None, 
                    time: datetime = None, worker_id: str = None):
        score = score or 0
        time = time or datetime.now()
        params = _Params(param_id, score, time)

        # Update best params for worker
        if worker_id not in self._worker_to_best_params or \
            score > self._worker_to_best_params[worker_id].score:
            self._worker_to_best_params[worker_id] = params
        
        # Update recent params for worker
        if worker_id not in self._worker_to_recent_params or \
            time > self._worker_to_recent_params[worker_id].time:
            self._worker_to_recent_params[worker_id] = params

    def get_params(self, shared_params: SharedParams, worker_id: str = None) -> Union[str, None]:
        if shared_params == SharedParams.NONE:
            return None
        elif shared_params == SharedParams.LOCAL_RECENT:
            return self._get_local_recent_params(worker_id)
        elif shared_params == SharedParams.LOCAL_BEST:
            return self._get_local_best_params(worker_id)
        elif shared_params == SharedParams.GLOBAL_RECENT:
            return self._get_global_recent_params()
        elif shared_params == SharedParams.GLOBAL_BEST:
            return self._get_global_best_params()
        else:
            raise ValueError('No such shared params type: "{}"'.format(shared_params))
    
    def _get_local_recent_params(self, worker_id):
        if worker_id not in self._worker_to_recent_params:
            return None
        
        params = self._worker_to_recent_params[worker_id]
        return params.param_id

    def _get_local_best_params(self, worker_id):
        if worker_id not in self._worker_to_best_params:
            return None
        
        params = self._worker_to_best_params[worker_id]
        return params.param_id

    def _get_global_recent_params(self):
        recent_params = [(params.time, params) for params in self._worker_to_recent_params.values()]
        if len(recent_params) == 0:
            return None

        recent_params.sort()
        (_, params) = recent_params[-1]
        return params.param_id

    def _get_global_best_params(self):
        best_params = [(params.score, params) for params in self._worker_to_best_params.values()]
        if len(best_params) == 0:
            return None

        best_params.sort()
        (_, params) = best_params[-1]
        return params.param_id

def _maybe_read_knobs_from_args(knob_config, args):
    parser = argparse.ArgumentParser()

    for (name, knob) in knob_config.items():
        if knob.value_type in [int, float, str]:
            parser.add_argument('--{}'.format(name), type=knob.value_type)
        elif knob.value_type in [list, bool]:
            parser.add_argument('--{}'.format(name), type=str)
        
    args_namespace = vars(parser.parse_known_args(args)[0])
    knobs_from_args = {}
    for (name, knob) in knob_config.items():
        if name in args_namespace and args_namespace[name] is not None:
            value = args_namespace[name]
            if knob.value_type in [list, bool]:
                value = eval(value)
            knobs_from_args[name] = value
            _info('Setting knob "{}" to be fixed value of "{}"...'.format(name, value))

    return knobs_from_args

def _check_dependencies(dependencies):
    for (dep, ver) in dependencies.items():
        # Warn that Keras models should additionally depend on TF for GPU usage
        if dep == ModelDependency.KERAS:
            _warn('Keras models can enable GPU usage with by adding a `tensorflow` dependency.')
        elif dep in [ModelDependency.TORCH, ModelDependency.TORCHVISION]:
            _info('PIP package `{}=={}` will be installed'.format(dep, ver))
        elif dep == ModelDependency.SCIKIT_LEARN:
            _info('PIP package `{}=={}` will be installed'.format(dep, ver))
        elif dep == ModelDependency.TENSORFLOW:
            # Warn that Keras models should additionally depend on TF for GPU usage
            _info('`tensorflow-gpu` of the same version will be installed if GPU is available during training.')
            _warn('TensorFlow models must cater for GPU-sharing with ' \
                    + '`config.gpu_options.allow_growth = True` (ref: https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth).')
        elif dep == ModelDependency.SINGA:
            _info('Conda packages `singa-gpu` or `singa-cpu` will be installed, depending on GPU availablility during training.')
        else:
            _info('PIP package `{}=={}` will be installed'.format(dep, ver))

def _check_model_class(py_model_class):
    if not issubclass(py_model_class, BaseModel):
        raise InvalidModelClassException('Model should extend `rafiki.model.BaseModel`')

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
        raise InvalidModelClassException('Static method `get_knob_config()` should return a dict[str, BaseKnob]')

    # Try serializing and deserialize knob config
    knob_config_bytes = serialize_knob_config(knob_config)
    knob_config = deserialize_knob_config(knob_config_bytes)

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