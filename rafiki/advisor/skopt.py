#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer
from collections import OrderedDict
import math
import numpy as np

from rafiki.model import CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob, PolicyKnob

from .constants import ParamsType, Proposal
from .advisor import BaseAdvisor, UnsupportedKnobError

FINAL_TRAIN_HOURS = 1 # No. of hours to conduct final train trials

class BayesOptAdvisor(BaseAdvisor):
    '''
        Performs standard hyperparameter tuning of models using Bayesian Optimization with Gaussian Processes.
    '''   
    @staticmethod
    def is_compatible(knob_config, budget):
        # Supports only CategoricalKnob, FixedKnob, IntegerKnob & FloatKnob
        return BaseAdvisor.has_only_knob_types(knob_config, [CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob, PolicyKnob])

    def __init__(self, knob_config, budget):
        super().__init__(knob_config, budget)
        (self._fixed_knob_config, knob_config) = self.extract_knob_type(knob_config, FixedKnob)
        (self._policy_knob_config, knob_config) = self.extract_knob_type(knob_config, PolicyKnob)
        self._dimensions = self._get_dimensions(knob_config)
        self._optimizer = self._make_optimizer(self._dimensions)
        self._search_results: (float, Proposal) = [] 

        # Prefer having certain policies
        if not self.has_policies(self.knob_config, ['EARLY_STOP']):
            print('To speed up hyperparameter search with Bayesian Optimization, having `EARLY_STOP` policy is preferred.')

    def propose(self, worker_id, trial_no):
        proposal_type = self._get_proposal_type(trial_no)
        meta = {'proposal_type': proposal_type}

        if proposal_type == 'SEARCH':
            knobs = self._propose_knobs(['EARLY_STOP'])
            return Proposal(trial_no, knobs, meta=meta)
        elif proposal_type == 'FINAL_TRAIN':
            knobs = self._propose_search_knobs()
            return Proposal(trial_no, knobs, meta=meta)
        elif proposal_type is None:
            return None

    def feedback(self, worker_id, result):
        proposal = result.proposal
        score = result.score
        proposal_type = proposal.meta.get('proposal_type') 
        knobs = proposal.knobs
        if score is None:
            return

        # Keep track of `SEARCH` trials' scores & proposals (for final train trials)
        if proposal_type == 'SEARCH':
            self._search_results.append((score, proposal))

        point = [knobs[name] for name in self._dimensions.keys()]
        self._optimizer.tell(point, -score)

    def _make_optimizer(self, dimensions):
        n_initial_points = 10
        return Optimizer(
            list(dimensions.values()),
            n_initial_points=n_initial_points,
            base_estimator='gp'
        )

    def _get_dimensions(self, knob_config):
        dimensions = OrderedDict({
            name: _knob_to_dimension(x)
                for (name, x)
                in knob_config.items()
        })
        return dimensions

    def _propose_knobs(self, policies=None):
        # Ask skopt
        point = self._optimizer.ask()
        knobs = { 
            name: _simplify_value(value) 
            for (name, value) 
            in zip(self._dimensions.keys(), point) 
        }

        # Add fixed & policy knobs
        knobs = self.merge_fixed_knobs(knobs, self._fixed_knob_config)
        knobs = self.merge_policy_knobs(knobs, self._policy_knob_config, policies or [])

        return knobs

    def _propose_search_knobs(self, policies=None):
        search_results = self._search_results
        # If no more search results, propose from model
        if len(search_results) == 0:
            return self._propose_knobs(policies)

        # Otherwise, determine best proposal and use it
        search_results.sort(key=lambda x: x[0])
        (score, proposal) = search_results.pop()
        knobs = proposal.knobs

        # Add policy knobs
        knobs = self.merge_policy_knobs(knobs, self._policy_knob_config, policies or [])

        return knobs

    def _get_proposal_type(self, trial_no):
        # If time's up, stop
        if self.get_train_hours_left() <= 0:
            return None

        # If trial's up, stop
        if self.get_trials_left(trial_no) <= 0:
            return None

        # If `EARLY_STOP` is not supported, just keep searching
        if not self.has_policies(self.knob_config, ['EARLY_STOP']):
            return 'SEARCH'

        # Schedule: |--<search>---||--<final train>--|
        # Check if final train trial
        # Otherwise, it is a search trial
        if self.get_train_hours_left() <= FINAL_TRAIN_HOURS:
            return 'FINAL_TRAIN'
        else:
            return 'SEARCH'

class BayesOptWithParamSharingAdvisor(BaseAdvisor):
    '''
        Performs hyperparameter tuning of models using Bayesian Optimization with Gaussian Processes,
        sharing globally best-scoring parameters in a greedy way.
    '''
    @staticmethod
    def is_compatible(knob_config, budget):
        # Supports only CategoricalKnob, FixedKnob, IntegerKnob & FloatKnob, and must have param sharing
        return BaseAdvisor.has_only_knob_types(knob_config, [CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob, PolicyKnob]) and \
            BaseAdvisor.has_policies(knob_config, ['SHARE_PARAMS'])

    def __init__(self, knob_config, budget):
        super().__init__(knob_config, budget)
        (self._fixed_knob_config, knob_config) = self.extract_knob_type(knob_config, FixedKnob)
        (self._policy_knob_config, knob_config) = self.extract_knob_type(knob_config, PolicyKnob)
        self._dimensions = self._get_dimensions(knob_config)
        self._optimizer = self._make_optimizer(self._dimensions)
        
        # Prefer having certain policies
        if not self.has_policies(self.knob_config, ['EARLY_STOP']):
            print('To speed up hyperparameter search with Bayesian Optimization, having `EARLY_STOP` policy is preferred.')

    def propose(self, worker_id, trial_no):
        proposal_type = self._get_proposal_type(trial_no)
        meta = {'proposal_type': proposal_type}

        if proposal_type == 'SEARCH':
            param = self._propose_param()
            knobs = self._propose_knobs(['SHARE_PARAMS', 'EARLY_STOP'])
            return Proposal(trial_no, knobs, params_type=param, meta=meta)
        elif proposal_type is None:
            return None

    def feedback(self, worker_id, result):
        proposal = result.proposal
        score = result.score
        knobs = proposal.knobs
        if score is None:
            return

        point = [ knobs[name] for name in self._dimensions.keys() ]
        self._optimizer.tell(point, -score)

    def _make_optimizer(self, dimensions):
        n_initial_points = 10
        return Optimizer(
            list(dimensions.values()),
            n_initial_points=n_initial_points,
            base_estimator='gp'
        )

    def _get_dimensions(self, knob_config):
        dimensions = OrderedDict({
            name: _knob_to_dimension(x)
                for (name, x)
                in knob_config.items()
        })
        return dimensions

    def _propose_knobs(self, policies=None):
        # Ask skopt
        point = self._optimizer.ask()
        knobs = { 
            name: _simplify_value(value) 
            for (name, value) 
            in zip(self._dimensions.keys(), point) 
        }

        # Add fixed & policy knobs
        knobs = self.merge_fixed_knobs(knobs, self._fixed_knob_config)
        knobs = self.merge_policy_knobs(knobs, self._policy_knob_config, policies or [])

        return knobs

    def _propose_param(self):
        total_hours = self.total_train_hours 
        hours_spent = total_hours - self.get_train_hours_left()
        return _propose_exp_greedy_param(hours_spent, total_hours)

    def _get_proposal_type(self, trial_no):
        # If time's up, stop
        if self.get_train_hours_left() <= 0:
            return None

        # If trial's up, stop
        if self.get_trials_left(trial_no) <= 0:
            return None
            
        # Keep conducting search trials
        return 'SEARCH'
    
def _propose_exp_greedy_param(t, t_div):
    e = math.exp(-4 * t / t_div) # e ^ (-4x) => 1 -> 0 exponential decay
    # No params with decreasing probability
    if np.random.random() < e:
        return ParamsType.NONE
    else:
        return ParamsType.GLOBAL_BEST

def _knob_to_dimension(knob):
    if isinstance(knob, CategoricalKnob):
        return Categorical([x.value for x in knob.values])
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

def _simplify_value(value):
    if isinstance(value, (np.int64)):
        return int(value) 

    return value
