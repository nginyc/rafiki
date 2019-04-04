import abc
import numpy as np
import random
from collections import namedtuple
from typing import Union, Dict, List, Tuple
from enum import Enum

from rafiki.model import BaseKnob, IntegerKnob, CategoricalKnob, FloatKnob, \
                FixedKnob, ListKnob

class UnsupportedKnobConfigError(Exception): pass
class UnsupportedKnobError(Exception): pass

class AdvisorType(Enum):
    RANDOM = 'RANDOM'
    SKOPT = 'SKOPT'
    ENAS = 'ENAS'

ADVISOR_TYPES = [AdvisorType.SKOPT, AdvisorType.ENAS, AdvisorType.RANDOM]

def make_advisor(knob_config: Dict[str, BaseKnob], advisor_type=None, **kwargs):
    if advisor_type is None:
        for advisor_type in ADVISOR_TYPES:
            clazz = _get_advisor_class_from_type(advisor_type)
            if clazz.is_compatible(knob_config):
                return clazz(knob_config, **kwargs)
        
        raise UnsupportedKnobConfigError()
    else:
        advisor_type = AdvisorType(advisor_type)
        clazz = _get_advisor_class_from_type(advisor_type)
        assert clazz.is_compatible(knob_config)
        return clazz(knob_config, **kwargs)

def _get_advisor_class_from_type(advisor_type):
    if advisor_type == AdvisorType.ENAS:
        from .tf import EnasAdvisor
        return EnasAdvisor
    elif advisor_type == AdvisorType.SKOPT:
        from .skopt import SkoptAdvisor
        return SkoptAdvisor
    elif advisor_type == AdvisorType.RANDOM:
        return RandomAdvisor

class ParamsType(Enum):
    LOCAL_RECENT = 'LOCAL_RECENT'
    LOCAL_BEST = 'LOCAL_BEST'
    GLOBAL_RECENT = 'GLOBAL_RECENT'
    GLOBAL_BEST = 'GLOBAL_BEST'
    NONE = 'NONE'

class TrainStrategy(Enum):
    STANDARD = 'STANDARD' # Model should train to its maximum potential
    EARLY_STOP = 'EARLY_STOP' # Model should stop as early as possible
    NONE = 'NONE' # Model would not be trained

class Proposal():
    def __init__(self, 
                knobs: Dict[str, any], 
                params_type: ParamsType = ParamsType.NONE, # Parameters to use for this trial
                is_valid = True, # If a trial is invalid, the worker will sleep for a while before trying again
                train_strategy: TrainStrategy = TrainStrategy.STANDARD, # How should the model train
                should_evaluate = True, # Whether this trial should be evaluated
                should_save_to_disk = True): # Whether this trial's trained model should be saved to disk
        self.knobs = knobs
        self.params_type = ParamsType(params_type)
        self.is_valid = is_valid
        self.train_strategy = TrainStrategy(train_strategy)
        self.should_evaluate = should_evaluate
        self.should_save_to_disk = should_save_to_disk

    def to_jsonable(self):
        return {
            **self.__dict__,
            'params_type': self.params_type.value,
            'train_strategy': self.train_strategy.value
        }

    @staticmethod
    def from_jsonable(jsonable):
        return Proposal(**jsonable)

class BaseAdvisor(abc.ABC):
    '''
    Base advisor class for knobs
    '''   
    @staticmethod
    @abc.abstractmethod
    def is_compatible(knob_config: Dict[str, BaseKnob]) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(self, knob_config: Dict[str, BaseKnob], **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def propose(self, trial_no: int = None, total_trials: int = None, 
                concurrent_trial_nos: List[int] = []) -> Proposal:
        '''
        :param int trial_no: Upcoming trial no to get proposal for
        :param int total_trials: Total no. of trials for this instance of tuning
        :param list[int] concurrent_trial_nos: Trial nos of other trials that are currently concurrently running 
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, score: float, proposal: Proposal):
        raise NotImplementedError()

class RandomAdvisor(BaseAdvisor):
    '''
    Advisor that randomly chooses knobs with no mathematical guarantee. 
    '''   
    @staticmethod
    def is_compatible(knob_config):
        # Compatible with all knobs
        return True

    def __init__(self, knob_config):
        self._knob_config = knob_config

    def propose(self, trial_no, total_trials, concurrent_trial_nos=[]):
        # Randomly propose knobs
        knobs = {
            name: self._propose_knob(knob) 
            for (name, knob) 
            in self._knob_config.items()
        }

        # Don't propose shared params
        params = ParamsType.NONE

        proposal = Proposal(knobs, params)
        return proposal 

    def _propose_knob(self, knob):
        u = random.uniform(0, 1)
        if isinstance(knob, FloatKnob):
            return knob.value_min + u * (knob.value_max - knob.value_min)
        elif isinstance(knob, IntegerKnob):
            return knob.value_min + int(u * (knob.value_max - knob.value_min + 1))
        elif isinstance(knob, CategoricalKnob):
            i = int(u * len(knob.values))
            return knob.values[i]
        elif isinstance(knob, FixedKnob):
            return knob.value
        elif isinstance(knob, ListKnob):
            return [self._propose_knob(knob.items[i]) for i in range(len(knob))]
        else:
            raise UnsupportedKnobError(knob.__class__)

    def feedback(self, score, proposal):
        # Ignore feedback - not relevant for a random advisor
        pass
