import os
import json
import abc
import traceback
import pickle
from rafiki.advisor import make_advisor
from rafiki.predictor import ensemble_predictions
from rafiki.constants import TaskType

from .dataset import DatasetUtils

class InvalidModelClassException(Exception): pass
class InvalidModelParamsException(Exception): pass

class ModelUtils(DatasetUtils):
    '''
    Collection of utility methods for model developers e.g. dataset loading
    '''   
    pass

class BaseModel(abc.ABC):
    '''
    Rafiki's base model class that Rafiki models should extend. 
    Rafiki models should implement all abstract methods according to their associated tasks' specifications.
    '''   

    def __init__(self):
        self.utils = ModelUtils()
        super().__init__()

    @abc.abstractmethod
    def get_knob_config(self):
        '''
        Return a dictionary defining this model's knob configuration 
        (i.e. list of knob names, their data types and their ranges).

        :returns: Dictionary defining this model's knob configuration 
        :rtype:
            ::

                {
                    'knobs': {
                        'hidden_layer_units': {
                            'type': 'int',
                            'range': [2, 128]
                        },
                        'epochs': {
                            'type': 'int',
                            'range': [1, 1]
                        },
                        'learning_rate': {
                            'type': 'float_exp',
                            'range': [1e-5, 1e-1]
                        },
                        'batch_size': {
                            'type': 'int_cat',
                            'values': [1, 2, 4, 8, 16, 32, 64, 128]
                        }
                    }
                }
            
        '''
        raise NotImplementedError()

    def init(self, knobs):
        '''
        Initialize the model with a dictionary of knob values. 
        These knob values will be chosen by Rafiki based on the model's knob config.

        :param knobs: Dictionary of knob values for this model instance
        :type knobs: dict[str, any]
        '''
        pass

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
        :rtype: dict[string, string]
        '''

        raise NotImplementedError()

    @abc.abstractmethod
    def load_parameters(self, params):
        '''
        Load a dictionary of model parameters into this model instance.
        This will be used for trained model deserialization within Rafiki.
        The model will be considered *trained* subsequently.

        :param params: Dictionary of model parameters
        :type params: dict[string, string]
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy(self):
        '''
        Destroy this model instance, closing any sessions or freeing any connections.
        No other methods will be called subsequently.
        '''
        pass


def validate_model_class(model_class, train_dataset_uri, test_dataset_uri, task,
                queries=[], knobs=None):
    '''
    Validates whether a model class is properly defined. 
    The model instance's methods will be called in an order similar to that in Rafiki.

    :param str train_dataset_uri: URI of the train dataset for testing the training of model
    :param str test_dataset_uri: URI of the test dataset for testing the evaluating of model
    :param str task: Task type of model
    :param list[any] queries: List of queries for testing predictions with the trained model
    :param knobs: Knobs to train the model with. If not specified, knobs from an advisor will be used
    :type knobs: dict[str, any]
    :returns: The trained model
    '''
    print('Testing instantiation of model...')
    model_inst = model_class()

    print('Checking deprecated methods...')
    if getattr(model_inst, 'get_predict_label_mapping', None) is not None:
        print('WARNING: `get_predict_label_mapping` has been deprecated!')

    print('Testing getting of model\'s knob config...')
    knob_config = model_inst.get_knob_config()

    if not isinstance(knob_config, dict):
        raise InvalidModelClassException('`get_knob_config()` should return a dict[str, any]')

    if 'knobs' not in knob_config:
        raise InvalidModelClassException('`knob_config` should have a \'knobs\' key')
    
    advisor = make_advisor(knob_config)

    if knobs is None:
        knobs = advisor.propose()

    print('Testing initialization of model...')
    print('Using knobs: {}'.format(knobs))
    model_inst.init(knobs)

    print('Testing training of model...')
    model_inst.train(train_dataset_uri)

    print('Testing evaluation of model...')
    score = model_inst.evaluate(test_dataset_uri)

    if not isinstance(score, float):
        raise InvalidModelClassException('`evaluate()` should return a float!')

    print('Score: {}'.format(score))

    print('Testing dumping of parameters of model...')
    parameters = model_inst.dump_parameters()

    if not isinstance(parameters, dict):
        raise InvalidModelClassException('`dump_parameters()` should return a dict[str, any]')

    try:
        # Model parameters are pickled and put into DB
        parameters = pickle.loads(pickle.dumps(parameters))
    except Exception:
        traceback.print_stack()
        raise InvalidModelClassException('`parameters` should be serializable by `pickle`.')

    print('Testing destroying of model...')
    model_inst.destroy()

    print('Testing loading of parameters of model...')
    model_inst = model_class()
    model_inst.init(knobs)
    model_inst.load_parameters(parameters)

    print('Testing predictions with model...')
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
    
    print('The model definition is valid!')

    return model_inst



