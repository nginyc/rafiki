import abc
import numpy as np
import random
from typing import Union, Dict

from rafiki.model import BaseKnob, IntegerKnob, CategoricalKnob, FloatKnob, \
                FixedKnob, ListKnob, DynamicListKnob, MetadataKnob, Metadata

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
    def feedback(self, score: float, params: dict):
        raise NotImplementedError() 

from .skopt import SkoptKnobAdvisor
from .tf import EnasKnobAdvisor

class Advisor():
    def __init__(self, total_trials: int, knob_config: Dict[str, BaseKnob]):
        self._trial_count = 0
        self._total_trials = total_trials
        self._knob_config = knob_config

        # Let skopt propose for these basic knobs
        self._skopt_knob_config = { name: knob for (name, knob) in knob_config.items() 
                            if type(knob) in [IntegerKnob, CategoricalKnob, FloatKnob] }

        if len(self._skopt_knob_config) > 0:
            self._skopt_knob_adv = SkoptKnobAdvisor()
            self._skopt_knob_adv.start(total_trials, self._skopt_knob_config)

        # Let ENAS propose for list knobs
        self._enas_knob_config = { name: knob for (name, knob) in knob_config.items() 
                            if type(knob) in [ListKnob] }

        if len(self._enas_knob_config) > 0:
            self._enas_knob_adv = EnasKnobAdvisor()
            self._enas_knob_adv.start(self._enas_knob_config)

        # Initialize fixed knobs
        self._fixed_knobs = { name: knob.value for (name, knob) in knob_config.items() 
                            if type(knob) in [FixedKnob] }

        # Note metadata knobs
        self._metadata_knobs = { name: knob.metadata for (name, knob) in knob_config.items() 
                            if type(knob) in [MetadataKnob] }

        # Use naive params advisor
        self._params_adv = NaiveParamAdvisor()

    @property
    def knob_config(self) -> Dict[str, BaseKnob]:
        return self._knob_config

    def propose(self) -> (Dict[str, any], Dict[str, any]):
        knobs = {}

        # Merge knobs from advisors
        if len(self._skopt_knob_config) > 0:
            skopt_knobs = self._skopt_knob_adv.propose()
            knobs.update(skopt_knobs)
        
        if len(self._enas_knob_config) > 0:
            enas_knobs = self._enas_knob_adv.propose()
            knobs.update(enas_knobs)
    
        # Merge fixed knobs in
        knobs.update(self._fixed_knobs)

        # Merge metadata knobs in
        for (name, metadata) in self._metadata_knobs.items():
            value = self._get_metadata_value(metadata)
            knobs[name] = value

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        # Propose params
        params = self._params_adv.propose()

        return (knobs, params)

    def feedback(self, score: float, knobs: Dict[str, any], params: Dict[str, any]):
        self._trial_count += 1

        # Feedback to skopt
        if len(self._skopt_knob_config) > 0:
            skopt_knobs = { name: knob for (name, knob) in knobs.items() if name in self._skopt_knob_config }
            self._skopt_knob_adv.feedback(score, skopt_knobs)

        # Feedback to ENAS
        if len(self._enas_knob_config) > 0:
            enas_knobs = { name: knob for (name, knob) in knobs.items() if name in self._enas_knob_config }
            self._enas_knob_adv.feedback(score, enas_knobs)

        # Feedback to params
        self._params_adv.feedback(score, params)

    def _simplify_value(self, value):
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            return float(value)

        return value

    def _get_metadata_value(self, metadata):
        if metadata == Metadata.TRIAL_COUNT:
            return self._trial_count
        elif metadata == Metadata.TOTAL_TRIALS:
            return self._total_trials
        else:
            raise ValueError('No such metadata: {}'.format(metadata))

class NaiveParamAdvisor(BaseParamAdvisor):
    def __init__(self):
        self._params = {}

    def propose(self):
        # Return most recent params
        if len(self._params) == 0:
            return {}

        return self._params

    def feedback(self, score, params):
        for (name, value) in params.items():
            self._params[name] = value

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

    def feedback(self, score, knobs):
        # Ignore feedback - not relevant for a random advisor
        pass
