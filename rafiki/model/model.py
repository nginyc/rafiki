import abc
import numpy as np
from enum import Enum
from typing import Union, Dict, Type, List

from .knob import BaseKnob

class SharedParams(Enum):
    LOCAL_RECENT = 'LOCAL_RECENT'
    LOCAL_BEST = 'LOCAL_BEST'
    GLOBAL_RECENT = 'GLOBAL_RECENT'
    GLOBAL_BEST = 'GLOBAL_BEST'
    NONE = 'NONE'

class TrialConfig():
    def __init__(self, 
                is_valid: bool = True, # If a trial is invalid, the worker will sleep for a while before trying again
                should_evaluate: bool = True, # Whether this trial should be evaluated
                should_save: bool = True, # Whether this trial's trained model should be saved
                shared_params: SharedParams = SharedParams.LOCAL_RECENT, # Shared parameters to use for this trial
                override_knobs: Dict[str, any] = {}): # These override values in knobs proposed for this trial

        self.is_valid = is_valid
        self.should_evaluate = should_evaluate
        self.shared_params = shared_params
        self.should_save = should_save
        self.override_knobs = override_knobs


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
    def get_knob_config() -> Dict[str, BaseKnob]:
        '''
        Return a dictionary defining this model class' knob configuration 
        (i.e. list of knob names, their data types and their ranges).

        :returns: Dictionary defining this model's knob configuration 
        :rtype: dict[str, rafiki.model.BaseKnob]
        '''
        raise NotImplementedError()

    @staticmethod
    def get_trial_config(trial_no: int, total_trials: int, concurrent_trial_nos: List[int]) -> TrialConfig:
        '''
        Returns the configuration for a specific trial identified by its number.
        Allows for declarative scheduling and configuration of trials. 

        :param int trial_no: Upcoming trial no to get configuration for 
        :param int total_trials: Total no. of trials for this instance of tuning
        :param list[int] concurrent_trial_nos: Trial nos of other trials that are currently concurrently running 
        :returns: Trial configuration for trial #`trial_no`
        :rtype: TrialConfig
        '''
        return TrialConfig()

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

    @abc.abstractmethod
    def train(self, dataset_uri: str):
        '''
        Train this model instance with given dataset and initialized knob values.
        Additionally, a dictionary of trained shared parameters from previous trials is passed.

        :param str dataset_uri: URI of the dataset in a format specified by the task
        :param dict shared_params: { <param_name>: <param_value> }
        '''
        raise NotImplementedError()

    # TODO: Allow configuration of other metrics
    @abc.abstractmethod
    def evaluate(self, dataset_uri: str) -> float:
        '''
        Evaluate this model instance with given dataset after training. 
        This will be called only when model is *trained*.

        :param str dataset_uri: URI of the dataset in a format specified by the task
        :returns: Accuracy as float from 0-1 on the dataset
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

    @abc.abstractmethod
    def save_parameters(self, params_dir: str):
        '''
        Saves the parameters of this model to a directory.
        This will be called only when model is *trained*.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load_parameters(self, params_dir: str):
        '''
        Loads the parameters of this model from a directory.
        The model will be considered *trained* subsequently.
        '''
        raise NotImplementedError()

    def get_shared_parameters(self) -> Union[None, Dict[str, np.array]]:
        '''
        Returns a dictionary of trained parameters to share with future trials, after the model has been *trained*.

        :returns: { <param_name>: <param_value> }
        :rtype: Union[None, Dict[str, np.array]]
        '''
        return None

    def set_shared_parameters(self, shared_params: Dict[str, np.array]):
        '''
        Sets this model's shared parameters as an *untrained* model

        :param shared_params: { <param_name>: <param_value> }
        :type shaed_params: Dict[str, np.array]
        '''
        pass