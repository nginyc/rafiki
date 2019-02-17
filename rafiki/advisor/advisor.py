import abc
import numpy as np
import random
from typing import Union

from .knob import IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, ListKnob, DynamicListKnob

class UnsupportedKnobTypeError(Exception): pass

class BaseKnobAdvisor(abc.ABC):
    '''
    Base advisor class for knobs
    '''   
    @abc.abstractmethod
    def start(self, knobs: dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def propose(self) -> list:
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, score: float, knobs: list):
        raise NotImplementedError()

class BaseParamAdvisor(abc.ABC):
    '''
    Base advisor class for params
    '''  
    @abc.abstractmethod
    def propose(self) -> Union[str, None]:
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, score: float, param_id: str):
        raise NotImplementedError() 

from .skopt import SkoptKnobAdvisor
from .tf import EnasKnobAdvisor

class Advisor():
    def start(self, knob_config: dict):
        self._knob_config = knob_config
        self._skopt_knob_adv = SkoptKnobAdvisor()
        self._enas_knob_adv = EnasKnobAdvisor()

        # Let skopt propose for these basic knobs
        self._skopt_knob_config = { name: knob for (name, knob) in knob_config.items() 
                            if type(knob) in [IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob] }

        self._skopt_knob_adv.start(self._skopt_knob_config)

        # Let ENAS propose for list knobs
        self._enas_knob_config = { name: knob for (name, knob) in knob_config.items() 
                            if type(knob) in [ListKnob] }
        self._enas_knob_adv.start(self._enas_knob_config)

        # Use naive params advisor
        self._params_adv = NaiveParamAdvisor()

    @property
    def knob_config(self) -> dict:
        return self._knob_config

    def propose(self) -> (dict, str):
        knobs = {}

        # Merge knobs from advisors
        skopt_knobs = self._skopt_knob_adv.propose()
        knobs.update(skopt_knobs)
        enas_knobs = self._enas_knob_adv.propose()
        knobs.update(enas_knobs)

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        # Propose params
        param_id = self._params_adv.propose()

        return (knobs, param_id)

    def feedback(self, knobs: dict, score: float, param_id: str = None):
        # Feedback to skopt
        skopt_knobs = { name: knob for (name, knob) in knobs.items() if name in self._skopt_knob_config }
        self._skopt_knob_adv.feedback(score, skopt_knobs)

        # Feedback to ENAS
        enas_knobs = { name: knob for (name, knob) in knobs.items() if name in self._enas_knob_config }
        self._enas_knob_adv.feedback(score, enas_knobs)

        # Feedback to params
        if param_id is not None:
            self._params_adv.feedback(score, param_id)

    def _simplify_value(self, value):
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            return float(value)

        return value

class NaiveParamAdvisor(BaseParamAdvisor):
    def __init__(self):
        self._param_scores = []

    def propose(self):
        # Return most recent params
        if len(self._param_scores) == 0:
            return None

        (score, param_id) = self._param_scores[-1]
        return param_id

    def feedback(self, score, param_id):
        self._param_scores.append((score, param_id))

class RandomKnobAdvisor(BaseKnobAdvisor):
    '''
    Advisor that randomly chooses knobs with no mathematical guarantee. 
    '''   
    def start(self, knob_config):
        self._knob_config = knob_config

    def propose(self):
        knobs = {
            name: self._propose(knob) 
            for (name, knob) 
            in self._knob_config.items()
        }
        return knobs
            
    def _propose(self, knob):
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
            return [self._propose(knob.items[i]) for i in range(len(knob))]
        elif isinstance(knob, DynamicListKnob):
            list_len = knob.len_min + int(u * (knob.len_max - knob.len_min + 1))
            return [self._propose(knob.items[i]) for i in range(list_len)]
        else:
            raise UnsupportedKnobTypeError(knob.__class__)

    def feedback(self, knobs, score):
        # Ignore feedback - no relevant for a random advisor
        pass
