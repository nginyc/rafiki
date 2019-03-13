import os
import json
import abc
import traceback
import uuid
import inspect
import argparse
import numpy as np
from typing import Union, Dict, Type

from rafiki.model import BaseModel, BaseKnob, serialize_knob_config, deserialize_knob_config, \
                        parse_model_install_command, load_model_class
from rafiki.constants import TaskType, ModelDependency
from rafiki.param_store import ParamStore
from rafiki.predictor import ensemble_predictions

from .advisor import Advisor

class InvalidModelClassException(Exception): pass

def tune_model(py_model_class: Type[BaseModel], train_dataset_uri: str, val_dataset_uri: str, 
                total_trials: int = 25, params_root_dir: str = 'params/', should_save: bool = True, 
                advisor: Advisor = None, to_read_args: bool = True) -> (Dict[str, any], str):
    '''
    Tunes a model on a given dataset in the current environment.

    :param BaseModel py_model_class: The Python class for the model
    :param str train_dataset_uri: URI of the train dataset for testing the training of model
    :param str val_dataset_uri: URI of the validation dataset for testing the evaluation of model
    :param int total_trials: Total number of trials to tune the model over
    :param str params_root_dir: Root folder path to create subfolders to save each trial's model parameters
    :param bool should_save: If model parameters should be saved
    :param Advisor advisor: A pre-created advisor to use for tuning the model
    :param bool to_read_args: Whether should system args be read to retrieve default values for `num_trials` and knobs
    :rtype: (dict, str)
    :returns: (<knobs for best trained model>, <params directory for best trained model>)
    '''

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

    # Configure shared params store
    param_store = ParamStore()
    
    # Variables to track over trials
    best_score = 0
    best_knobs = None
    best_model_params_dir = None
    session_id = str(uuid.uuid4()) # Session ID for params store

    # Setup model class
    print('Running model class setup...')
    py_model_class.setup()

    # For every trial
    for i in range(1, total_trials + 1):
        trial_id = str(uuid.uuid4())
        _print_header('Trial #{} (ID: "{}")'.format(i, trial_id))
        
        # Get valid proposal from advisor
        while True:
            (knobs, params) = advisor.propose()
            validated_knobs = py_model_class.validate_knobs(knobs)
            if validated_knobs is None:
                # Feedback to advisor that knobs are invalid with score of 0
                advisor.feedback(0, knobs, {})
                continue
                
            knobs = validated_knobs
            break
        knobs = { **knobs, **knobs_from_args } # Override knobs from args
        print('Advisor proposed knobs:', knobs)

        # Retrieve params from store
        if len(params) > 0:
            print('Advisor proposed {} params'.format(len(params)))
        params = param_store.retrieve_params(session_id, params)

        # Load model
        model_inst = py_model_class(**knobs)

        # Train model
        print('Training model...')
        model_inst.train(train_dataset_uri, params)

        trial_params = model_inst.get_shared_parameters() or {}
        if len(trial_params) > 0:
            print('Model produced {} shared parameters'.format(len(trial_params)))

        # Evaluate model
        print('Evaluating model...')
        score = model_inst.evaluate(val_dataset_uri)
        if not isinstance(score, float):
            raise InvalidModelClassException('`evaluate()` should return a float!')

        print('Score:', score)
            
        if should_save:
            print('Saving model parameters...')
            params_dir = os.path.join(params_root_dir, trial_id + '/')
            if not os.path.exists(params_dir):
                os.mkdir(params_dir)
            model_inst.save_parameters(params_dir)
            print('Model parameters saved in {}'.format(params_dir))
        else:
            params_dir = None

        # Update best model
        if score > best_score:
            _info('Best model so far! Beats previous best of score {}!'.format(best_score))
            best_model_params_dir = params_dir
            best_knobs = knobs
            best_score = score

        # Feedback to advisor
        trial_params = param_store.store_params(session_id, trial_params, prefix=trial_id)
        advisor.feedback(score, knobs, trial_params)
    
    # Teardown model class
    print('Running model class teardown...')
    py_model_class.teardown()
    
    return (best_knobs, best_model_params_dir)

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

def _maybe_read_knobs_from_args(knob_config, args):
    parser = argparse.ArgumentParser()

    for (name, knob) in knob_config.items():
        if knob.value_type in [int, float, str]:
            parser.add_argument('--{}'.format(name), type=knob.value_type)
        elif knob.value_type in [list]:
            parser.add_argument('--{}'.format(name), type=str)
        
    args_namespace = vars(parser.parse_known_args(args)[0])
    knobs_from_args = {}
    for (name, knob) in knob_config.items():
        if name in args_namespace and args_namespace[name] is not None:
            value = args_namespace[name]
            if knob.value_type in [list]:
                value = json.loads(value)
            knobs_from_args[name] = value
            _info('Setting knob "{}" to be fixed value of "{}"...'.format(name, value))

    return knobs_from_args

def _check_dependencies(dependencies):
    for (dep, ver) in dependencies.items():
        # Warn that Keras models should additionally depend on TF for GPU usage
        if dep == ModelDependency.KERAS:
            _warn('Keras models can enable GPU usage with by adding a `tensorflow` dependency.')
        elif dep == ModelDependency.PYTORCH:
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