import abc
import numpy as np
from enum import Enum
from typing import Union, Dict, Type, List

from .knob import BaseKnob

KnobConfig = Dict[str, BaseKnob]
Params = Dict[str, Union[str, np.ndarray]]

class BaseModel(abc.ABC):
    '''
    Rafiki's base model class that Rafiki models should extend. 
    Rafiki models should implement all abstract methods according to their associated tasks' specifications,
    together with the static methods ``get_knob_config()`` (optional).

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
    @abc.abstractmethod
    def get_knob_config() -> KnobConfig:
        '''
        Return a dictionary defining this model class' knob configuration 
        (i.e. list of knob names, their data types and their ranges).

        :returns: Dictionary defining this model's knob configuration 
        :rtype: KnobConfig
        '''
        raise NotImplementedError()

    @staticmethod
    def setup():
        '''
        Runs class-wide setup logic (e.g. initialize a graph/session shared across trials).
        '''
        pass

    @staticmethod
    def teardown():
        '''
        Runs class-wide teardown logic (e.g. closes a session shared across trials).
        '''
        pass

    # TODO: Improve doc
    @abc.abstractmethod
    def train(self, dataset_path: str, shared_params: Params = None):
        '''
        Train this model instance with given dataset and initialized knob values.
        Additionally, a dictionary of trained parameters shared from previous trials could be passed.

        :param str dataset_path: File path of the train dataset file in the local file system, in a format specified by the task
        :param Params shared_params: { <param_name>: <param_value> }
        '''
        raise NotImplementedError()

    # TODO: Allow configuration of other metrics
    @abc.abstractmethod
    def evaluate(self, dataset_path: str) -> float:
        '''
        Evaluate this model instance with given dataset after training. 
        This will be called only when model is *trained*.

        :param str dataset_path: File path of the validation dataset file in the local file system, in a format specified by the task
        :returns: Accuracy as float from 0-1 on the validation dataset
        :rtype: float
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, queries: list) -> list:
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

    # TODO: Document format of params
    @abc.abstractmethod
    def dump_parameters(self) -> Params:
        '''
        Returns a dictionary of model parameters for persistence and to share with future trials during training.
        This dictionary should be serializable by the Python's ``pickle`` module.
        This will be used for trained model serialization within Rafiki.
        This will be called only when model is *trained*.

        :returns: { <param_name>: <param_value> }
        :rtype: Params
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load_parameters(self, params: Params):
        '''
        Loads previously trained parameters for this model. This model's initialized knob values would match those during training.
        The model will be considered *trained* subsequently.
        :param Params params: { <param_name>: <param_value> }
        '''
        raise NotImplementedError()
    
    def destroy(self):
        '''
        Destroy this model instance, freeing any resources held by this model instance.
        No other instance methods will be called subsequently.
        '''
        pass 