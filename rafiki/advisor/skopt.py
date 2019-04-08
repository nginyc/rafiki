from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer
from collections import OrderedDict
from enum import Enum
import math
import numpy as np

from rafiki.model import CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob

from .advisor import BaseAdvisor, UnsupportedKnobError, Proposal, ParamsType, TrainStrategy

class ParamPolicy(Enum):
    NONE = 'NONE'
    GREEDY = 'GREEDY' # Always wants global best
    LINEAR_GREEDY = 'LINEAR_GREEDY' # (1 - p) probability of wanting global best, p decreases linearly
    EXP_GREEDY = 'EXP_GREEDY' # (1 - p) probability of wanting global best, p decays exponentially

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

    def __init__(self, knob_config, param_policy=ParamPolicy.EXP_GREEDY):
        (self._fixed_knobs, knob_config) = _extract_fixed_knobs(knob_config)
        self._dimensions = self._get_dimensions(knob_config)
        self._optimizer = self._make_optimizer(self._dimensions)
        self._param_policy = ParamPolicy(param_policy)
        self._recent_feedback = [] # [(score, proposal)]

    def propose(self, worker_id, trial_no, total_trials, concurrent_trial_nos=[]):
        trial_type = self._get_trial_type(trial_no, total_trials, concurrent_trial_nos)

        if trial_type is None:
            return Proposal({}, is_valid=False)
        elif trial_type == 'SEARCH':
            param = self._propose_param(trial_no, total_trials)
            knobs = self._propose_knobs()
            return Proposal(knobs, 
                            train_strategy=TrainStrategy.STOP_EARLY,
                            params_type=param)
        elif trial_type == 'FINAL_TRAIN':
            knobs = self._propose_best_recent_knobs()
            return Proposal(knobs,
                            train_strategy=TrainStrategy.STANDARD,
                            params_type=ParamsType.NONE)

    def feedback(self, score, proposal):
        num_sample_trials = 10
        knobs = proposal.knobs

        # Keep track of last X trials' knobs & scores (for final train trials)
        self._recent_feedback = [(score, proposal), *self._recent_feedback[:(num_sample_trials - 1)]]

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

    def _propose_knobs(self):
        # Ask skopt
        point = self._optimizer.ask()
        knobs = { 
            name: value 
            for (name, value) 
            in zip(self._dimensions.keys(), point) 
        }

        # Add fixed knobs
        knobs = { **self._fixed_knobs, **knobs }
        return knobs

    def _propose_best_recent_knobs(self):
        recent_feedback = self._recent_feedback
        # If hasn't collected feedback, propose from model
        if len(recent_feedback) == 0:
            return self._propose_knobs()

        # Otherwise, determine best recent proposal and use it
        (score, proposal) = sorted(recent_feedback)[-1]

        return proposal.knobs

    def _propose_param(self, trial_no, total_trials):
        policy = self._param_policy
        if policy == ParamPolicy.NONE:
            return ParamsType.NONE
        elif policy == ParamPolicy.GREEDY:
            return ParamsType.GLOBAL_BEST
        elif policy == ParamPolicy.EXP_GREEDY:
            return self._propose_exp_greedy_param(trial_no, total_trials)
        elif policy == ParamPolicy.LINEAR_GREEDY:
            return self._propose_linear_greedy_param(trial_no, total_trials)

    def _get_trial_type(self, trial_no, total_trials, concurrent_trial_nos):
        num_final_train_trials = 1
        policy = self._param_policy

        # If param policy is param sharing
        if policy in [ParamPolicy.EXP_GREEDY, ParamPolicy.LINEAR_GREEDY]:
            # Keep conducting search trials
            return 'SEARCH'
        
        # Otherwise, there is no param sharing
        # Schedule: |--<search>---||--<final train>--|

        # Check if final train trial
        if trial_no > total_trials - num_final_train_trials:
            # Wait for all search trials to finish 
            if self._if_preceding_trials_are_running(trial_no, concurrent_trial_nos):
                return None

            return 'FINAL_TRAIN'

        # Otherwise, it is a search trial
        return 'SEARCH'
        
    def _propose_exp_greedy_param(self, trial_no, total_trials):
        t = trial_no
        t_div = total_trials
        e = math.exp(-4 * t / t_div) # e ^ (-4x) => 1 -> 0 exponential decay
        # No params with decreasing probability
        if np.random.random() < e:
            return ParamsType.NONE
        else:
            return ParamsType.GLOBAL_BEST

    def _propose_linear_greedy_param(self, trial_no, total_trials):
        t = trial_no
        t_div = total_trials
        e = 0.5 - 0.5 * t / t_div # 0.5 -> 0 linearly
        # No params with decreasing probability
        if np.random.random() < e:
            return ParamsType.NONE
        else:
            return ParamsType.GLOBAL_BEST

    def _if_preceding_trials_are_running(self, trial_no, concurrent_trial_nos):
        if len(concurrent_trial_nos) == 0:
            return False

        min_trial_no = min(concurrent_trial_nos)
        return min_trial_no < trial_no


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
