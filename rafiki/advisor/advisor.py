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

import abc
import random
from typing import List, Type, Union
from datetime import datetime, timedelta

from rafiki.constants import Budget, BudgetOption
from rafiki.model import IntegerKnob, CategoricalKnob, FloatKnob, ArchKnob, \
                        FixedKnob, PolicyKnob, KnobConfig, BaseKnob

from .constants import AdvisorType, Proposal, TrialResult

DEFAULT_TRAIN_HOURS = 0.1
DEFAULT_MAX_TRIALS = -1

class UnsupportedKnobConfigError(Exception): pass
class UnsupportedKnobError(Exception): pass

# Advisor to use, in descending priority
ADVISOR_TYPES = [AdvisorType.FIXED, 
                AdvisorType.BAYES_OPT_WITH_PARAM_SHARING, 
                AdvisorType.BAYES_OPT,
                AdvisorType.ENAS, 
                AdvisorType.RANDOM]

def make_advisor(knob_config: KnobConfig, budget: Budget):
    for advisor_type in ADVISOR_TYPES:
        clazz = _get_advisor_class_from_type(advisor_type)
        if clazz.is_compatible(knob_config, budget):
            return clazz(knob_config, budget)
    
    raise UnsupportedKnobConfigError()

def _get_advisor_class_from_type(advisor_type):
    if advisor_type == AdvisorType.ENAS:
        from .tf import EnasAdvisor
        return EnasAdvisor
    elif advisor_type == AdvisorType.BAYES_OPT:
        from .skopt import BayesOptAdvisor
        return BayesOptAdvisor
    elif advisor_type == AdvisorType.BAYES_OPT_WITH_PARAM_SHARING:
        from .skopt import BayesOptWithParamSharingAdvisor
        return BayesOptWithParamSharingAdvisor
    elif advisor_type == AdvisorType.RANDOM:
        return RandomAdvisor
    elif advisor_type == AdvisorType.FIXED:
        return FixedAdvisor


class BaseAdvisor(abc.ABC):
    '''
    Base advisor class
    '''   
    @staticmethod
    @abc.abstractmethod
    def is_compatible(knob_config: KnobConfig, budget: Budget) -> bool:
        raise NotImplementedError()
    
    def __init__(self, knob_config: KnobConfig, budget: BudgetOption):
        self.knob_config = knob_config
        self.total_train_hours = float(budget.get(BudgetOption.TIME_HOURS, DEFAULT_TRAIN_HOURS))
        self.max_trials = budget.get(BudgetOption.MODEL_TRIAL_COUNT, DEFAULT_MAX_TRIALS)

        # Keep track of time budget
        self._start_time = datetime.now()
        self._stop_time = self._start_time + timedelta(hours=self.total_train_hours)

    @abc.abstractmethod
    def propose(self, worker_id: str, trial_no: int) -> Union[Proposal, None]:
        '''
        Returns a proposal for the worker, or None if the tuning process is to be stopped. 

        :param str worker_id: Worker to make a proposal for
        :param int trial_no: Trial number associated with this proposal
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, worker_id: str, result: TrialResult):
        '''
        Ingests feedback for the result of a proposal.
        '''
        raise NotImplementedError()

    # Returns no. of hours left for training based on allocated budget
    def get_train_hours_left(self) -> float:
        time_left = self._stop_time - datetime.now()
        return time_left.total_seconds() / (60 * 60)

    # Returns no. of trials left for training based on allocated budget (excluding current trial)
    def get_trials_left(self, trial_no) -> int:
        if self.max_trials < 0:
            return 9999999
        return self.max_trials - trial_no + 1

    # Helps detect presence of policies in knob config
    @staticmethod
    def has_policies(knob_config: KnobConfig, policies: List[str]):
        avail_policies = [x.policy for (name, x) in knob_config.items() if isinstance(x, PolicyKnob)]
        for policy in policies:
            if policy not in avail_policies:
                return False
        
        return True

    # Helps detect presence of knob types in knob config
    @staticmethod
    def has_only_knob_types(knob_config: KnobConfig, knob_types: List[Type[BaseKnob]]):
        for (name, knob) in knob_config.items():
            if not isinstance(knob, tuple(knob_types)):
                return False
        
        return True
    
    # Helps extraction of a certain type of knob from knob config
    @staticmethod
    def extract_knob_type(knob_config, knob_type):
        sub_knob_config = {name: knob for (name, knob) in knob_config.items() if isinstance(knob, knob_type)}
        knob_config = {name: knob for (name, knob) in knob_config.items() if not isinstance(knob, knob_type)}
        return (sub_knob_config, knob_config)

    # Merge fixed knobs into `knobs`
    @staticmethod
    def merge_fixed_knobs(knobs, fixed_knob_config):
        return {**knobs, **{name: x.value.value for (name, x) in fixed_knob_config.items()}}

    # Merge policy knobs into `knobs`, activating `policies`
    @staticmethod
    def merge_policy_knobs(knobs, policy_knob_config, policies):
        policies = policies or []
        policy_knobs = {name: (True if x.policy in policies else False) for (name, x) in policy_knob_config.items()}
        return {**knobs, **policy_knobs} 


class FixedAdvisor(BaseAdvisor):
    '''
    Advisor that runs a single trial
    ''' 
    @staticmethod
    def is_compatible(knob_config, budget):
        # Must only have fixed & policy knobs
        return BaseAdvisor.has_only_knob_types(knob_config, [FixedKnob, PolicyKnob])

    def propose(self, worker_id, trial_no):
        if trial_no > 1:
            return None

        # Propose fixed knob values
        knobs = {name: knob.value.value for (name, knob) in self.knob_config.items()}

        proposal = Proposal(trial_no, knobs)
        return proposal 

    def feedback(self, worker_id, result):
        # Ignore feedback
        pass


class RandomAdvisor(BaseAdvisor):
    '''
    Advisor that uniformly randomly chooses knobs. 
    '''   
    @staticmethod
    def is_compatible(knob_config, budget):
        # Compatible with all knobs
        return True

    def propose(self, worker_id, trial_no):
        # If time's up, stop
        if self.get_train_hours_left() <= 0:
            return None

        # If trial's up, stop
        if self.get_trials_left(trial_no) <= 0:
            return None 

        # Randomly propose knobs
        knobs = {
            name: self._propose_knob(knob) 
            for (name, knob) 
            in self.knob_config.items()
        }

        proposal = Proposal(trial_no, knobs)
        return proposal 

    def _propose_knob(self, knob):
        if isinstance(knob, FloatKnob):
            u = random.uniform(0, 1)
            return knob.value_min + u * (knob.value_max - knob.value_min)
        elif isinstance(knob, IntegerKnob):
            u = random.uniform(0, 1)
            return knob.value_min + int(u * (knob.value_max - knob.value_min + 1))
        elif isinstance(knob, CategoricalKnob):
            u = random.uniform(0, 1)
            i = int(u * len(knob.values))
            return knob.values[i].value
        elif isinstance(knob, FixedKnob):
            return knob.value.value
        elif isinstance(knob, ArchKnob):
            knob_value = []
            for values in knob.items:
                u = random.uniform(0, 1)
                i = int(u * len(values))
                knob_value.append(values[i].value)
            return knob_value
        elif isinstance(knob, PolicyKnob):
            return False
        else:
            raise UnsupportedKnobError(knob.__class__)

    def feedback(self, worker_id, result):
        # Ignore feedback - not relevant for a random advisor
        pass
