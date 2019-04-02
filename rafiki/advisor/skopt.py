from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer
from collections import OrderedDict
from enum import Enum
import numpy as np

from rafiki.model import CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob

from .advisor import BaseAdvisor, UnsupportedKnobError, Proposal, ParamsType

class ParamPolicy(Enum):
    NONE = 'NONE'
    LOCAL_RECENT = 'LOCAL_RECENT'
    LINEAR_GREEDY = 'LINEAR_GREEDY'
    EXP_GREEDY = 'EXP_GREEDY'

class SkoptAdvisor(BaseAdvisor):
    '''
    Performs general hyperparameter tuning of models using Bayesian Optimization 
        using Gaussian Processes as implemented by `skopt`.
    '''   
    @staticmethod
    def is_compatible(knob_config):
        # Supports CategoricalKnob, FixedKnob, IntegerKnob and FloatKnob
        for (name, knob) in knob_config.items():
            if not isinstance(knob, (CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob)):
                return False

        return True

    def __init__(self, knob_config, param_policy=ParamPolicy.LINEAR_GREEDY):
        (self._fixed_knobs, knob_config) = _extract_fixed_knobs(knob_config)
        self._dimensions = self._get_dimensions(knob_config)
        self._optimizer = self._make_optimizer(self._dimensions)
        self._param_policy = ParamPolicy(param_policy)

    def propose(self, trial_no, total_trials, concurrent_trial_nos=[]):
        # Ask skopt
        point = self._optimizer.ask()
        knobs = { 
            name: value 
            for (name, value) 
            in zip(self._dimensions.keys(), point) 
        }

        # Add fixed knobs
        knobs = { **self._fixed_knobs, **knobs }

        params = self._propose_param(trial_no, total_trials)
        
        proposal = Proposal(knobs, params)

        return proposal

    def feedback(self, score, proposal):
        knobs = proposal.knobs
        point = [ knobs[name] for name in self._dimensions.keys() ]
        self._optimizer.tell(point, -score)

    def _make_optimizer(self, dimensions):
        n_random_starts = 10
        return Optimizer(
            list(dimensions.values()),
            n_random_starts=n_random_starts,
            base_estimator='gp'
        )

    def _get_dimensions(self, knob_config):
        dimensions = OrderedDict({
            name: _knob_to_dimension(x)
                for (name, x)
                in knob_config.items()
        })
        return dimensions

    def _propose_param(self, trial_no, total_trials):
        policy = self._param_policy
        if policy == ParamPolicy.NONE:
            return ParamsType.NONE
        elif policy == ParamPolicy.LOCAL_RECENT:
            return ParamsType.LOCAL_RECENT
        elif policy == ParamPolicy.EXP_GREEDY:
            return self._propose_exp_greedy_param(trial_no, total_trials)
        elif policy == ParamPolicy.LINEAR_GREEDY:
            return self._propose_linear_greedy_param(trial_no, total_trials)
        
    def _propose_exp_greedy_param(self, trial_no, total_trials):
        t = trial_no
        t_div = total_trials
        e_base = 0.99
        e = pow(e_base, 500 * t / t_div) # 0.99 -> 0.0065
        # No params with decreasing probability
        if np.random.random() < e:
            return ParamsType.NONE
        else:
            return ParamsType.GLOBAL_BEST

    def _propose_linear_greedy_param(self, trial_no, total_trials):
        t = trial_no
        t_div = total_trials
        e = 1 - t / t_div
        # No params with decreasing probability
        if np.random.random() < e:
            return ParamsType.NONE
        else:
            return ParamsType.GLOBAL_BEST

def _knob_to_dimension(knob):
    if isinstance(knob, CategoricalKnob):
        return Categorical(knob.values)
    elif isinstance(knob, IntegerKnob):
        return Integer(knob.value_min, knob.value_max)
    elif isinstance(knob, FloatKnob):
        if knob.is_exp:
            # Avoid error in skopt when low/high are 0
            value_min = knob.value_min if knob.value_min != 0 else 1e-12
            value_max = knob.value_max if knob.value_max != 0 else 1e-12
            return Real(value_min, value_max, 'log-uniform')
        else:
            return Real(knob.value_min, knob.value_max, 'uniform')
    else:
        raise UnsupportedKnobError(knob.__class__)

def _unzeroify(self, value):
    if value == 0:
        return 1e-9

def _extract_fixed_knobs(knob_config):
    fixed_knobs = { name: knob.value for (name, knob) in knob_config.items() if isinstance(knob, FixedKnob) }
    knob_config = { name: knob for (name, knob) in knob_config.items() if not isinstance(knob, FixedKnob) }
    return (fixed_knobs, knob_config)
