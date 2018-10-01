import os
import abc
import pickle
from importlib import import_module

TEMP_MODEL_FILE_NAME = 'temp'

class InvalidModelParamsException(Exception):
    pass

class BaseModel(abc.ABC):
    '''
    Rafiki's base model class that Rafiki models should extend. 
    Rafiki models should implement all abstract methods according to their associated tasks' specifications.
    '''   

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

def load_model_class(model_file_bytes, model_class):
    # Save the model file to disk
    f = open('{}.py'.format(TEMP_MODEL_FILE_NAME), 'wb')
    f.write(model_file_bytes)
    f.close()

    # Import model file as module
    mod = import_module(TEMP_MODEL_FILE_NAME)

    # Extract model class from module
    clazz = getattr(mod, model_class)

    # Remove temporary file
    os.remove(f.name)

    return clazz