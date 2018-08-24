import abc
import pickle
import os

class InvalidModelParamsException(Exception):
    pass

class BaseModel(abc.ABC):    
    
    @abc.abstractmethod
    def get_hyperparameter_config(self):
        raise NotImplementedError()


    def init(self, hyperparameters):
        '''
            hyperparameters - dict<hyperparameter_name: String, hyperparameter_value: Any>
        '''

    @abc.abstractmethod
    def train(self, dataset_config):
        '''
            dataset_config - dict that configures the dataset for training

            Supported Dataset Types are task-dependent
        '''
        raise NotImplementedError()

    # TODO: Allow configuration of other metrics
    @abc.abstractmethod
    def evaluate(self, dataset_config):
        '''
            dataset_config - dict that configures the dataset for evaluation

            Returns:
                float as accuracy on test dataset

            Supported Dataset Types are task-dependent
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, queries):
        '''
            queries - iterable of Query Type as queries

            Returns:
                iterable of Prediction Type as predictions for the corresponding queries
                
            Supported Query Types are task-dependent
            Prediction Type is task-dependent
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def dump_parameters(self):
        '''
            Returns:
                dict<parameter_name: String, parameter_value: String> as all model parameters
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load_parameters(self, params):
        '''
            params - dict<parameter_name: String, parameter_value: String>
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy(self):
        pass