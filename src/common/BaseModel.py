import abc
import pickle
import os

class InvalidModelParamsException(Exception):
    pass

class BaseModel(abc.ABC):    

    def __init__(self):
        self.hyperparameters = None

    
    @abc.abstractmethod
    def get_hyperparameter_config(self):
        raise NotImplementedError()


    def init(self, hyperparameters):
        '''
            hyperparameters - dict<hyperparameter_name: String, hyperparameter_value: Any>
        '''
        self.hyperparameters = hyperparameters
        

    @abc.abstractmethod
    def train(self, dataset_config):
        '''
            dataset_config - instance of DatasetConfig that configures the dataset for training
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, dataset_config):
        '''
            dataset_config - instance of DatasetConfig that configures the dataset for evaluation

            Returns:
                float as accuracy on test dataset
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, queries):
        '''
            queries - iterable of Any as queries

            Returns:
                iterable of int as labels for the corresponding queries
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def dump_parameters(self):
        '''
            Returns:
                dict<parameter_name: String, parameter_value: string> as all model parameters
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load_parameters(self, params):
        '''
            params - dict<parameter_name: String, parameter_value: string>
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy(self):
        pass