import abc
import numpy as np
import random
import logging
from typing import Union, Dict

from rafiki.model import BaseKnob, IntegerKnob, CategoricalKnob, FloatKnob, \
                FixedKnob, ListKnob, SharedParams

logger = logging.getLogger(__name__)

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

class Advisor():
    def __init__(self, knob_config: Dict[str, BaseKnob]):
        self._knob_config = knob_config

        # Let skopt propose for these basic knobs
        self._skopt_knob_config = { name: knob for (name, knob) in knob_config.items() 
                            if type(knob) in [IntegerKnob, CategoricalKnob, FloatKnob] }

        if len(self._skopt_knob_config) > 0:
            from .skopt import SkoptKnobAdvisor
            self._skopt_knob_adv = SkoptKnobAdvisor()
            self._skopt_knob_adv.start(self._skopt_knob_config)

        # Let ENAS propose for list knobs
        self._enas_knob_config = { name: knob for (name, knob) in knob_config.items() 
                            if type(knob) in [ListKnob] }

        if len(self._enas_knob_config) > 0:
            from .tf import EnasKnobAdvisor
            self._enas_knob_adv = EnasKnobAdvisor()
            self._enas_knob_adv.start(self._enas_knob_config)

        # Initialize fixed knobs
        self._fixed_knobs = { name: knob.value for (name, knob) in knob_config.items() 
                            if type(knob) in [FixedKnob] }

    @property
    def knob_config(self) -> Dict[str, BaseKnob]:
        return self._knob_config

    def propose(self) -> Dict[str, any]:
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

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        logger.info('Proposing knobs {}...'.format(knobs))
        return knobs

    def feedback(self, score: Union[float, None], knobs: Dict[str, any]):
        logger.info('Received feedback of score {} for knobs {}'.format(score, knobs))

        # Feedback to skopt
        if len(self._skopt_knob_config) > 0:
            skopt_knobs = { name: knob for (name, knob) in knobs.items() if name in self._skopt_knob_config }
            self._skopt_knob_adv.feedback(score, skopt_knobs)

        # Feedback to ENAS
        if len(self._enas_knob_config) > 0:
            enas_knobs = { name: knob for (name, knob) in knobs.items() if name in self._enas_knob_config }
            self._enas_knob_adv.feedback(score, enas_knobs)

    def _simplify_value(self, value):
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            return float(value)

        return value

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
        else:
            raise UnsupportedKnobTypeError(knob.__class__)

    def feedback(self, score, knobs):
        # Ignore feedback - not relevant for a random advisor
        pass

# class EpsilonGreedyParamAdvisor(BaseParamAdvisor):
#     def __init__(self, base_epsilon=0.5, trial_div=25):
#         self._base_epsilon = base_epsilon
#         self._trial_div = trial_div
#         self._trial_count = 0
#         self._worker_to_params: Dict[str, Dict[str, _Param]] = {}
#         self._best_params: Dict[str, _Param] = {}

#     def propose(self, worker_id):
#         t = self._trial_count
#         t_div = self._trial_div
#         e_base = self._base_epsilon
#         e = self._compute_epsilon(t, t_div, e_base)

#         # Use current worker's params with decreasing probability
#         if np.random.random() < e:
#             prop_params = self._worker_to_params.get(worker_id, {})
#         # Otherwise, use best params across workers
#         else:
#             prop_params = self._best_params

#         return { name: param.value for (name, param) in prop_params.items() }

#     def feedback(self, score, params, worker_id):
#         # Override worker's params (use most recent)
#         worker_params = self._worker_to_params.get(worker_id, {})
#         for (name, value) in params.items():
#             worker_params[name] = _Param(value, score)
#         self._worker_to_params[worker_id] = worker_params

#         # For each param has better score than the best so far, replace it
#         for (name, value) in params.items():
#             if name not in self._best_params or \
#                 score > self._best_params[name].score:
#                 self._best_params[name] = _Param(value, score)

#         self._trial_count += 1

#     def _compute_epsilon(self, t, t_div, e_base):
#         e = pow(e_base, 1 + t / t_div)
#         return e
