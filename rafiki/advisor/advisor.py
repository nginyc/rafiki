import abc
import random
from typing import List, Type, Dict
from enum import Enum
from datetime import datetime, timedelta

from rafiki.model import IntegerKnob, CategoricalKnob, FloatKnob, ArchKnob, \
                        FixedKnob, PolicyKnob, KnobConfig, Knobs, BaseKnob
from rafiki.param_cache import ParamsType

DEFAULT_TRAIN_HOURS = 0.1
DEFAULT_MAX_TRIALS = -1

BUDGET_OPTIONS = ['TIME_HOURS', 'GPU_COUNT', 'MODEL_TRIAL_COUNT'] 
Budget = Dict[str, any]

class UnsupportedKnobConfigError(Exception): pass
class UnsupportedKnobError(Exception): pass

class AdvisorType(Enum):
    FIXED = 'FIXED'
    BAYES_OPT_WITH_PARAM_SHARING = 'BAYES_OPT_WITH_PARAM_SHARING'
    BAYES_OPT = 'BAYES_OPT'
    RANDOM = 'RANDOM'
    ENAS = 'ENAS'

class TrainWorker():
    def __init__(self, 
                worker_id: str, 
                gpus: int = 0): # No. of GPUs allocated to worker
        self.worker_id = worker_id
        self.gpus = gpus

    def __str__(self):
        return str(self.__dict__)

class Proposal():
    def __init__(self, 
                knobs: Knobs = None, 
                worker_id: str = None, # Worker to run this proposal
                params_type: ParamsType = ParamsType.NONE, # Parameters to use for this trial
                to_eval=True, # Whether the model should be evaluated
                to_cache_params=False, # Whether this trial's parameters should be cached
                to_save_params=True, # Whether this trial's parameters should be persisted
                meta: dict = None): # Extra metadata associated with proposal
        self.knobs = knobs
        self.worker_id = worker_id
        self.params_type = ParamsType(params_type)
        self.to_eval = to_eval
        self.to_cache_params = to_cache_params
        self.to_save_params = to_save_params
        self.meta = meta or {}

    def __str__(self):
        return str(self.__dict__)

class ProposalResult():
    def __init__(self, 
                proposal: Proposal, 
                score: float, # Score for the proposal
                worker_id: str): # ID of worker that ran the proposal
        self.proposal = proposal
        self.score = score
        self.worker_id = worker_id

    def __str__(self):
        return str(self.__dict__)


# Advisor to use, in descending priority
ADVISOR_TYPES = [AdvisorType.FIXED, 
                AdvisorType.ENAS, 
                AdvisorType.BAYES_OPT_WITH_PARAM_SHARING, 
                AdvisorType.BAYES_OPT,
                AdvisorType.RANDOM]

def make_advisor(knob_config: KnobConfig, budget: Budget, workers: List[TrainWorker]):
    for advisor_type in ADVISOR_TYPES:
        clazz = _get_advisor_class_from_type(advisor_type)
        if clazz.is_compatible(knob_config, budget):
            return clazz(knob_config, budget, workers)
    
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
    Base advisor class for knobs
    '''   
    @staticmethod
    @abc.abstractmethod
    def is_compatible(knob_config: KnobConfig, budget: Budget) -> bool:
        raise NotImplementedError()
    
    def __init__(self, knob_config: KnobConfig, budget: Budget, workers: List[TrainWorker]):
        self.knob_config = knob_config
        self.workers = workers
        self.total_train_hours = budget.get('TIME_HOURS', DEFAULT_TRAIN_HOURS)
        self.max_trials = budget.get('MODEL_TRIAL_COUNT', DEFAULT_MAX_TRIALS)

        # Keep track of time budget
        self._start_time = datetime.now()
        self._stop_time = self._start_time + timedelta(hours=self.total_train_hours)

    @abc.abstractmethod
    def propose(self, worker_id: str, num_trials: int) -> Proposal:
        '''
        Returns a proposal or None if there are currently no proposals. 

        :param str worker_id: Worker to make a proposal for
        :param int num_trials: Total no. of trials that has been started
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, result: ProposalResult):
        '''
        Ingests feedback for the result of a proposal.
        '''
        raise NotImplementedError()

    # Returns no. of hours left for training based on allocated budget
    def get_train_hours_left(self) -> float:
        time_left = self._stop_time - datetime.now()
        return time_left.total_seconds() / (60 * 60)

    # Returns no. of trials left for training based on allocated budget (excluding current trial)
    def get_trials_left(self, num_trials) -> int:
        if self.max_trials < 0:
            return 9999999
        return self.max_trials - num_trials

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
        # Must only have fixed knobs
        return BaseAdvisor.has_only_knob_types(knob_config, [FixedKnob])

    def propose(self, worker_id, num_trials):
        if num_trials >= 1:
            return None

        # Propose fixed knob values
        knobs = {name: knob.value.value for (name, knob) in self.knob_config.items()}

        proposal = Proposal(knobs)
        return proposal 

    def feedback(self, result):
        # Ignore feedback
        pass


class RandomAdvisor(BaseAdvisor):
    '''
    Advisor that randomly chooses knobs with no mathematical guarantee. 
    '''   
    @staticmethod
    def is_compatible(knob_config, budget):
        # Compatible with all knobs
        return True

    def propose(self, worker_id, num_trials):
        # If time's up, stop
        if self.get_train_hours_left() <= 0:
            return None

        # If trial's up, stop
        if self.get_trials_left(num_trials) <= 0:
            return None 

        # Randomly propose knobs
        knobs = {
            name: self._propose_knob(knob) 
            for (name, knob) 
            in self.knob_config.items()
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
            return knob.values[i].value
        elif isinstance(knob, FixedKnob):
            return knob.value.value
        elif isinstance(knob, ArchKnob):
            knob_value = []
            for values in knob.items:
                i = int(u * len(values))
                knob_value.append(values[i].value)
            return knob_value
        elif isinstance(knob, PolicyKnob):
            return False
        else:
            raise UnsupportedKnobError(knob.__class__)

    def feedback(self, result):
        # Ignore feedback - not relevant for a random advisor
        pass
