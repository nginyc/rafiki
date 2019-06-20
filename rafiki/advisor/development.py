import os
import json
import traceback
import uuid
import inspect
import argparse
import time
from typing import Dict, Type

from rafiki.model import BaseModel, BaseKnob, serialize_knob_config, deserialize_knob_config, \
                        parse_model_install_command, load_model_class
from rafiki.constants import ModelDependency
from rafiki.param_store import ParamStore
from rafiki.predictor import ensemble_predictions

from .advisor import make_advisor, ParamsType

# TODO: Better doc
def tune_model(py_model_class: Type[BaseModel], train_dataset_path: str, val_dataset_path: str, 
                test_dataset_path: str = None, budget: Dict[str, any] = None, 
                params_root_dir: str = 'params/') -> (Dict[str, any], float, str):
    '''
    Tunes a model on a given dataset in the current environment.

    Additionally, reads knob values and budget options from CLI arguments.

    :param BaseModel py_model_class: The Python class for the model
    :param str train_dataset_path: File path of the train dataset for training of the model
    :param str val_dataset_path: File path of the validation dataset for evaluating trained models
    :param str test_dataset_path: URI of the validation dataset for testing the final best trained model, if provided
    :param str params_root_dir: Root folder path to create subfolders to save each trial's model parameters
    :rtype: (dict, float, str)
    :returns: (<knobs for best model>, <test score for best model>, <params directory for best model>)
    ''' 
    # Note start time
    start_time = time.time()

    # Retrieve config of model
    _print_header('Checking model configuration...')
    knob_config = py_model_class.get_knob_config()
    _check_knob_config(knob_config)

    # Read knob values from CLI args
    _print_header('Starting trials...')
    knobs_from_args = _maybe_read_knobs_from_args(knob_config)

    # Read budget options from CLI args
    budget_from_args = _maybe_read_budget_from_args()
    budget = {**(budget or {}), **budget_from_args}
    inform_user(f'Using budget {budget}...')

    # Make advisor
    advisor = make_advisor(knob_config, budget)
    inform_user(f'Using advisor "{type(advisor).__name__}"...')

    # Create params store
    param_store = ParamStore()
    
    # Variables to track over trials
    best_model_score = 0
    best_trial_no = 0 
    best_model_test_score = None
    best_proposal = None
    best_model_params_file_path = None

    # Until a proposal telling worker to stop, keep conducting trials
    trial_no = 0
    while True:
        trial_id = str(uuid.uuid4())
        trial_no += 1

        # Get proposal from advisor, overriding knobs from args & trial config
        proposal = advisor.propose('localhost', trial_no, [])
        print('Proposal from advisor:', proposal)
        if proposal.should_stop:
            print('Advisor proposed to stop training')
            break
        _assert_jsonable(proposal.to_jsonable())
        assert not proposal.should_wait

        _print_header('Trial #{} (ID: "{}")'.format(trial_no, trial_id))
        
        proposal.knobs = { **proposal.knobs, **knobs_from_args } 
        print('Advisor proposed knobs:', proposal.knobs)
        if proposal.params_type != ParamsType.NONE:
            print('Advisor proposed params:', proposal.params_type.name)
        if not proposal.should_eval:
            print('Advisor proposed whether to evaluate:', proposal.should_eval)

        # Load model
        model_inst = py_model_class(**proposal.knobs)

        shared_params = None
        if proposal.params_type != ParamsType.NONE:
            print('Retrieving shared params...')
            shared_params = param_store.retrieve_params(proposal.params_type)

        print('Training model...')
        model_inst.train(train_dataset_path, shared_params=shared_params)
        trial_params = model_inst.dump_parameters()
        if trial_params:
            print('Model produced {} params'.format(len(trial_params)))

        # Evaluate model
        score = None
        if proposal.should_eval:
            print('Evaluating model...')
            score = model_inst.evaluate(val_dataset_path)
            if not isinstance(score, float):
                raise Exception('`evaluate()` should return a float!')

            print('Score on validation dataset:', score)

        if score is not None:
            # Feedback to advisor
            print('Giving feedback to advisor...')
            advisor.feedback(score, proposal)

            if proposal.should_save_to_disk:
                # Save model
                print('Saving trained model to disk...')
                params_bytes = ParamStore.serialize_params(trial_params)
                params_file_path = os.path.join(params_root_dir, '{}.model'.format(trial_id))
                with open(params_file_path, 'wb') as f:
                    f.write(params_bytes)
                inform_user('Model saved to {}'.format(params_file_path))


                # Update best saved model
                if score > best_model_score and proposal.should_save_to_disk:
                    inform_user('Best saved model so far! Beats previous best of score {}!'.format(best_model_score))
                        
                    best_model_params_file_path = params_file_path
                    best_proposal = proposal
                    best_model_score = score
                    best_trial_no = trial_no

                    # Test best model, if test dataset provided
                    if test_dataset_path is not None:
                        print('Evaluting new best model on test dataset...')
                        best_model_test_score = model_inst.evaluate(test_dataset_path)
                        inform_user('Score on test dataset: {}'.format(best_model_test_score))
                 

        # Update params store
        if trial_params:
            print('Storing trial\'s params...')
            param_store.store_params(trial_params, score)

        # Destroy model
        model_inst.destroy()
    
    # Declare best model
    if best_proposal is not None:
        inform_user('Best trial #{} has knobs {} with score of {}'.format(best_trial_no, best_proposal.knobs, best_model_score))
        if best_model_test_score is not None:
            inform_user('...with test score of {}'.format(best_model_test_score))
        if best_model_params_file_path is not None:
            inform_user('...saved at {}'.format(best_model_params_file_path)) 
        
    # Teardown model class
    print('Running model class teardown...')
    py_model_class.teardown()

    # Print duration
    duration = time.time() - start_time
    print('Tuning took a total of {}s'.format(duration))

    return (best_proposal, best_model_test_score, best_model_params_file_path)


# TODO: Fix method, more thorough testing of model API
def test_model_class(model_file_path: str, model_class: str, task: str, 
                    dependencies: Dict[str, str], queries: list = None, **kwargs):
    '''
    Tests whether a model class is properly defined by running a full train-inference flow.
    The model instance's methods will be called in an order similar to that in Rafiki.

    Refer to `tune_model` for additional parameters to be passed.

    :param str model_file_path: Path to a single Python file that contains the definition for the model class
    :param str model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
    :param str task: Task type of model
    :param dict[str, str] dependencies: Model's dependencies
    :param str train_dataset_path: File path of the train dataset for training of the model
    :param str val_dataset_path: File path of the validation dataset for evaluating trained models
    :param list[any] queries: List of queries for testing predictions with the trained model
    :returns: The trained model
    '''
    _print_header('Installing & checking model dependencies...')
    _check_dependencies(dependencies)

    _print_header('Checking loading of model & model definition...')
    with open(model_file_path, 'rb') as f:
        model_file_bytes = f.read()
    py_model_class = load_model_class(model_file_bytes, model_class, temp_mod_name=model_class)
    _check_model_class(py_model_class)

    # Simulation of training process
    (proposal, _, params_file_path) = tune_model(py_model_class, **kwargs)
   
    # Load best trained model's parameters
    model_inst = None
    if proposal is not None:
        _print_header('Checking loading of parameters from disk...')
        model_inst = py_model_class(**proposal.knobs)
        with open(params_file_path, 'rb') as f:
            params_bytes = f.read()
        params = ParamStore.deserialize_params(params_bytes)
        model_inst.load_parameters(params)

        if queries is not None:
            _print_header('Checking predictions...')
            print('Using queries: {}'.format(queries))
            predictions = model_inst.predict(queries)

            # Verify format of predictions
            for prediction in predictions:
                _assert_jsonable(prediction, Exception('Each `prediction` should be JSON serializable'))

            # Ensemble predictions in predictor
            predictions = ensemble_predictions([predictions], task)

            print('Predictions: {}'.format(predictions))

    py_model_class.teardown()

    inform_user('No errors encountered while testing model!')

    return model_inst

def warn_user(msg):
    print(f'\033[93mWARNING: {msg}\033[0m')

def inform_user(msg):
    print(f'\033[94m{msg}\033[0m')

def _maybe_read_knobs_from_args(knob_config):
    parser = argparse.ArgumentParser()

    for (name, knob) in knob_config.items():
        if knob.value_type in [int, float, str]:
            parser.add_argument('--{}'.format(name), type=knob.value_type)
        elif knob.value_type in [list, bool]:
            parser.add_argument('--{}'.format(name), type=str)
        
    args_namespace = vars(parser.parse_known_args()[0])
    knobs_from_args = {}
    for (name, knob) in knob_config.items():
        if name in args_namespace and args_namespace[name] is not None:
            value = args_namespace[name]
            if knob.value_type in [list, bool]:
                value = eval(value)
            knobs_from_args[name] = value
            inform_user('Setting knob "{}" to be fixed value of "{}"...'.format(name, value))

    return knobs_from_args

def _maybe_read_budget_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_COUNT', type=int, default=0)
    parser.add_argument('--TIME_HOURS', type=float, default=0.1) # 6 min
    budget_from_args = vars(parser.parse_known_args()[0])
    return budget_from_args

def _check_model_class(py_model_class):
    if not issubclass(py_model_class, BaseModel):
        raise Exception('Model should extend `rafiki.model.BaseModel`')

    if inspect.isfunction(getattr(py_model_class, 'init', None)):
        warn_user('`init` has been deprecated - use `__init__` for your model\'s initialization logic instead')

    if inspect.isfunction(getattr(py_model_class, 'get_knob_config', None)) and \
        not isinstance(py_model_class.__dict__.get('get_knob_config', None), staticmethod):
        warn_user('`get_knob_config` has been changed to a `@staticmethod`')


def _check_dependencies(dependencies):
    if not isinstance(dependencies, dict):
        raise Exception('`dependencies` should be a dict[str, str]')

    for (dep, ver) in dependencies.items():
        if dep == ModelDependency.KERAS:
            # Warn that Keras models should additionally depend on TF for GPU usage
            inform_user('Keras models can enable GPU usage with by adding a `tensorflow` dependency.')
        elif dep in [ModelDependency.TORCH, ModelDependency.TORCHVISION]:
            pass
        elif dep == ModelDependency.SCIKIT_LEARN:
            pass
        elif dep == ModelDependency.TENSORFLOW:
            warn_user('TensorFlow models must cater for GPU-sharing with ' \
                    + '`config.gpu_options.allow_growth = True` (ref: https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth).')
        elif dep == ModelDependency.SINGA:
            pass
    
    install_command = parse_model_install_command(dependencies, enable_gpu=False)
    install_command_with_gpu = parse_model_install_command(dependencies, enable_gpu=True)
    inform_user(f'Install command (without GPU): `{install_command}`')
    inform_user(f'Install command (with GPU): `{install_command_with_gpu}`')


def _check_knob_config(knob_config):
    if not isinstance(knob_config, dict) or \
        any([(not isinstance(name, str) or not isinstance(knob, BaseKnob)) for (name, knob) in knob_config.items()]):
        raise Exception('Static method `get_knob_config()` should return a dict[str, BaseKnob]')

    # Try serializing and deserialize knob config
    knob_config_bytes = serialize_knob_config(knob_config)
    knob_config = deserialize_knob_config(knob_config_bytes)

def _assert_jsonable(jsonable, exception=None):
    try:
        json.dumps(jsonable)
    except Exception as e:
        traceback.print_stack()
        raise exception or e 

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

def _print_header(msg):
    print('-' * (len(msg) + 4))
    print('| {} |'.format(msg))
    print('-' * (len(msg) + 4))
