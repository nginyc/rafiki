import abc
import numpy as np
import random
from collections import namedtuple
import logging
from typing import Union, Dict

from rafiki.model import BaseKnob, IntegerKnob, CategoricalKnob, FloatKnob, \
                FixedKnob, ListKnob, MetadataKnob, Metadata

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

class BaseParamAdvisor(abc.ABC):
    '''
    Base advisor class for params
    '''  
    @abc.abstractmethod
    def propose(self, worker_id: str) -> Union[str, None]:
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, score: float, params: dict, worker_id: str):
        raise NotImplementedError()

class Advisor():
    def __init__(self, knob_config: Dict[str, BaseKnob]):
        self._trial_count = 0
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

        # Note metadata knobs
        self._metadata_knobs = { name: knob.metadata for (name, knob) in knob_config.items() 
                            if type(knob) in [MetadataKnob] }

        self._params_adv = self._get_param_advisor()

    @property
    def knob_config(self) -> Dict[str, BaseKnob]:
        return self._knob_config

    def propose(self, worker_id: str = None) -> (Dict[str, any], Dict[str, any]):
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
            value = self._get_metadata_value(metadata, worker_id)
            knobs[name] = value

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        # Propose params
        params = self._params_adv.propose(worker_id)

        logger.info('Proposing to worker of ID "{}" knobs {} with {} shared params...'.format(worker_id, knobs, len(params)))
        return (knobs, params)

    def feedback(self, score: float, knobs: Dict[str, any], params: Dict[str, any], worker_id: str = None):
        logger.info('Received feedback of score {} for knobs {} with {} shared params'
                    .format(score, knobs, len(params)))

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
        self._params_adv.feedback(score, params, worker_id)
    
    def _get_param_advisor(self):
        return RecentParamAdvisor()

    def _simplify_value(self, value):
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            return float(value)

        return value

    def _get_metadata_value(self, metadata, worker_id):
        if metadata == Metadata.TRIAL_COUNT:
            return self._trial_count
        elif metadata == Metadata.WORKER_ID:
            return worker_id
        else:
            raise ValueError('No such metadata: {}'.format(metadata))

class RecentParamAdvisor(BaseParamAdvisor):
    def __init__(self):
        self._params = {}

    def propose(self, worker_id):
        # Return most recent params
        return self._params

    def feedback(self, score, params, worker_id):
        for (name, value) in params.items():
            self._params[name] = value

_Param = namedtuple('_Param', ('value', 'score'))

class EpsilonGreedyParamAdvisor(BaseParamAdvisor):
    def __init__(self, base_epsilon=0.5, trial_div=25):
        self._base_epsilon = base_epsilon
        self._trial_div = trial_div
        self._trial_count = 0
        self._worker_to_params: Dict[str, Dict[str, _Param]] = {}
        self._best_params: Dict[str, _Param] = {}

    def propose(self, worker_id):
        t = self._trial_count
        t_div = self._trial_div
        e_base = self._base_epsilon
        e = self._compute_epsilon(t, t_div, e_base)

        # Use current worker's params with decreasing probability
        if np.random.random() < e:
            prop_params = self._worker_to_params.get(worker_id, {})
        # Otherwise, use best params across workers
        else:
            prop_params = self._best_params

        return { name: param.value for (name, param) in prop_params.items() }

    def feedback(self, score, params, worker_id):
        # Override worker's params (use most recent)
        worker_params = self._worker_to_params.get(worker_id, {})
        for (name, value) in params.items():
            worker_params[name] = _Param(value, score)
        self._worker_to_params[worker_id] = worker_params

        # For each param has better score than the best so far, replace it
        for (name, value) in params.items():
            if name not in self._best_params or \
                score > self._best_params[name].score:
                self._best_params[name] = _Param(value, score)

        self._trial_count += 1

    def _compute_epsilon(self, t, t_div, e_base):
        e = pow(e_base, 1 + t / t_div)
        return e

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
