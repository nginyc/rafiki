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
import traceback
import inspect
import argparse
import time
from datetime import datetime
from typing import Dict, Type, List, Any

from rafiki.constants import ModelDependency, Budget
from rafiki.advisor import ParamsType, Proposal, TrialResult, make_advisor
from rafiki.predictor import get_ensemble_method, Query, Prediction
from rafiki.param_store import FileParamStore, ParamStore
from rafiki.redis import ParamCache, TrainCache, InferenceCache

from .model import BaseModel, BaseKnob, Params
from .utils import serialize_knob_config, deserialize_knob_config, parse_model_install_command, load_model_class
                    
def tune_model(py_model_class: Type[BaseModel], train_dataset_path: str, val_dataset_path: str, 
                test_dataset_path: str = None, budget: Budget = None, 
                train_args: Dict[str, any] = None) -> (Dict[str, Any], float, Params):

    worker_id = 'local'

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

    # Create caches & stores
    param_store: ParamStore = FileParamStore()
    param_cache: ParamCache = ParamCache()
    train_cache: TrainCache = TrainCache()
    
    # Variables to track over trials
    best_model_score = -1
    best_trial_no = 0 
    best_model_test_score = None
    best_proposal = None
    best_store_params_id = None

    # Train worker tells advisor that it is free
    train_cache.add_worker(worker_id)

    # Until there's no more proposals, keep conducting trials
    trial_no = 0
    while True:
        trial_no += 1

        # Advisor checks free workers
        worker_ids = train_cache.get_workers()
        assert worker_id in worker_ids

        # Advisor checks worker doesn't already have a proposal
        proposal = train_cache.get_proposal(worker_id)
        assert proposal is None

        # Advisor sends a proposal to worker
        # Overriding knobs from args
        proposal = advisor.propose(worker_id, trial_no)
        if proposal is None:
            print('No more proposals from advisor - to stop training')
            break
        proposal.knobs = { **proposal.knobs, **knobs_from_args } 
        train_cache.create_proposal(worker_id, proposal)

        # Worker receives proposal
        proposal = train_cache.get_proposal(worker_id)
        assert proposal is not None

        # Worker starts trial
        _print_header(f'Trial #{trial_no}')
        print('Proposal from advisor:', proposal)

        # Worker loads model
        model_inst = py_model_class(**proposal.knobs)

        # Worker pulls shared params
        shared_params = _pull_shared_params(proposal, param_cache)

        # Worker trains model
        print('Training model...')
        model_inst.train(train_dataset_path, shared_params=shared_params, **(train_args or {}))

        # Worker evaluates model
        result = _evaluate_model(model_inst, proposal, val_dataset_path)

        # Worker caches/saves model parameters
        store_params_id = _save_model(model_inst, proposal, result, param_cache, param_store)

        # Update best saved model
        if result.score is not None and store_params_id is not None and result.score > best_model_score:
            inform_user('Best saved model so far! Beats previous best of score {}!'.format(best_model_score))
            best_store_params_id = store_params_id
            best_proposal = proposal
            best_model_score = result.score
            best_trial_no = trial_no

            # Test best model, if test dataset provided
            if test_dataset_path is not None:
                print('Evaluating new best model on test dataset...')
                best_model_test_score = model_inst.evaluate(test_dataset_path)
                inform_user('Score on test dataset: {}'.format(best_model_test_score))

        # Worker sends result to advisor
        print('Giving feedback to advisor...')
        train_cache.create_result(worker_id, result) 
        train_cache.delete_proposal(worker_id)

        # Advisor receives result
        # Advisor ingests feedback
        result = train_cache.take_result(worker_id)
        assert result is not None
        advisor.feedback(worker_id, result)

        # Destroy model
        model_inst.destroy()
    
    # Train worker tells advisor that it is no longer free
    train_cache.delete_worker(worker_id)

    # Declare best model
    if best_proposal is not None:
        inform_user('Best trial #{} has knobs {} with score of {}'.format(best_trial_no, best_proposal.knobs, best_model_score))
        if best_model_test_score is not None:
            inform_user('...with test score of {}'.format(best_model_test_score))

    # Load params for best model
    best_params = None
    if best_store_params_id is not None:
        best_params = param_store.load(best_store_params_id)

    # Teardown model class
    print('Running model class teardown...')
    py_model_class.teardown()

    # Print duration
    duration = time.time() - start_time
    print('Tuning took a total of {}s'.format(duration))

    return (best_proposal, best_model_test_score, best_params)

def make_predictions(queries: List[Any], task: str, py_model_class: Type[BaseModel], proposal: Proposal, params: Params) -> List[Any]:
    inference_cache: InferenceCache = InferenceCache()
    worker_id = 'local'

    print('Queries: {}'.format(queries))

    # Worker load best trained model's parameters
    model_inst = None
    _print_header('Loading trained model...')
    model_inst = py_model_class(**proposal.knobs)
    model_inst.load_parameters(params)

    # Inference worker tells predictor that it is free
    inference_cache.add_worker(worker_id)

    # Predictor receives queries
    queries = [Query(x) for x in queries]

    # Predictor checks free workers
    worker_ids = inference_cache.get_workers()
    assert worker_id in worker_ids

    # Predictor sends query to worker
    inference_cache.add_queries_for_worker(worker_id, queries)

    # Worker receives query
    queries_at_worker = inference_cache.pop_queries_for_worker(worker_id, len(queries))
    assert len(queries_at_worker) == len(queries)

    # Worker makes prediction on queries
    _print_header('Making predictions with trained model...')
    predictions = model_inst.predict([x.query for x in queries_at_worker])
    predictions = [Prediction(x, query.id, worker_id) for (x, query) in zip(predictions, queries_at_worker)]

    # Worker sends predictions to predictor
    inference_cache.add_predictions_for_worker(worker_id, predictions)

    # Predictor receives predictions
    predictions_at_predictor = []
    for query in queries:
        prediction = inference_cache.take_prediction_for_worker(worker_id, query.id)
        assert prediction is not None
        predictions_at_predictor.append(prediction)

    # Predictor ensembles predictions
    ensemble_method = get_ensemble_method(task)
    print(f'Ensemble method: {ensemble_method}')
    out_predictions = []
    for prediction in predictions_at_predictor:
        prediction = prediction.prediction
        _assert_jsonable(prediction, Exception('Each `prediction` should be JSON serializable'))
        out_prediction = ensemble_method([prediction])
        out_predictions.append(out_prediction)

    print('Predictions: {}'.format(out_predictions))

    return (out_predictions, model_inst)

# TODO: Fix method, more thorough testing of model API
def test_model_class(model_file_path: str, model_class: str, task: str, dependencies: Dict[str, str], 
                    train_dataset_path: str, val_dataset_path: str, test_dataset_path: str = None, 
                    budget: Budget = None, train_args: Dict[str, any] = None, queries: List[Any] = None) -> (List[Any], BaseModel):
    '''
    Tests whether a model class is *more likely* to be correctly defined by *locally* simulating a full train-inference flow on your model
    on a given dataset. The model's methods will be called in an manner similar to that in Rafiki.

    This method assumes that your model's Python dependencies have already been installed. 
    
    This method also reads knob values and budget options from CLI arguments. 
    For example, you can pass e.g. ``--TIME_HOURS=0.01`` to configure the budget, or ``--learning_rate=0.01`` to fix a knob's value.

    :param model_file_path: Path to a single Python file that contains the definition for the model class
    :param model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
    :param task: Task type of model
    :param dependencies: Model's dependencies
    :param train_dataset_path: File path of the train dataset for training of the model
    :param val_dataset_path: File path of the validation dataset for evaluating trained models
    :param test_dataset_path: File path of the test dataset for testing the final best trained model, if provided
    :param budget: Budget for model training
    :param train_args: Additional arguments to pass to models during training, if any
    :param queries: List of queries for testing predictions with the trained model
    :returns: (<predictions of best trained model>, <best trained model>)

    '''
    _print_header('Installing & checking model dependencies...')
    _check_dependencies(dependencies)

    _print_header('Checking loading of model & model definition...')
    with open(model_file_path, 'rb') as f:
        model_file_bytes = f.read()
    py_model_class = load_model_class(model_file_bytes, model_class, temp_mod_name=model_class)
    _check_model_class(py_model_class)

    # Simulation of training
    (best_proposal, _, best_params) = tune_model(py_model_class, train_dataset_path, val_dataset_path, 
                                                test_dataset_path=test_dataset_path, budget=budget, 
                                                train_args=train_args)

    # Simulation of inference
    model_inst = None
    predictions = None
    if best_proposal is not None and best_params is not None and queries is not None:
        (predictions, model_inst) = make_predictions(queries, task, py_model_class, best_proposal, best_params)

    py_model_class.teardown()

    inform_user('No errors encountered while testing model!')

    return (predictions, model_inst)

def warn_user(msg):
    print(f'\033[93mWARNING: {msg}\033[0m')

def inform_user(msg):
    print(f'\033[94m{msg}\033[0m')

def _pull_shared_params(proposal: Proposal, param_cache: ParamCache):
    if proposal.params_type == ParamsType.NONE:
        return None

    print('Retrieving shared params from cache...')
    shared_params = param_cache.retrieve_params(proposal.params_type)
    return shared_params

def _evaluate_model(model_inst: BaseModel, proposal: Proposal, 
                    val_dataset_path: str) -> TrialResult:
    if not proposal.to_eval: 
        return TrialResult(proposal)
        
    print('Evaluating model...')
    score = model_inst.evaluate(val_dataset_path)

    if not isinstance(score, float):
        raise Exception('`evaluate()` should return a float!')
        
    print('Score on validation dataset:', score)
    return TrialResult(proposal, score=score)

def _save_model(model_inst: BaseModel, proposal: Proposal, result: TrialResult, 
                param_cache: ParamCache, param_store: ParamStore):
    if not proposal.to_cache_params and not proposal.to_save_params:
        return None
    
    print('Dumping model parameters...')
    params = model_inst.dump_parameters()
    if proposal.to_cache_params:
        print('Storing shared params in cache...')
        param_cache.store_params(params, score=result.score, time=datetime.now())
    
    store_params_id = None
    if proposal.to_save_params:
        print('Saving shared params...')
        store_params_id = param_store.save(params)

    return store_params_id   

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
    parser.add_argument('--TIME_HOURS', type=float, default=0.01) # < 1 min
    parser.add_argument('--MODEL_TRIAL_COUNT', type=int, default=-1)
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
