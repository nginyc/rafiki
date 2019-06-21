from enum import Enum
from typing import Dict

from rafiki.model import Knobs

BUDGET_OPTIONS = ['TIME_HOURS', 'GPU_COUNT', 'MODEL_TRIAL_COUNT'] 
Budget = Dict[str, any]

class AdvisorType(Enum):
    FIXED = 'FIXED'
    BAYES_OPT_WITH_PARAM_SHARING = 'BAYES_OPT_WITH_PARAM_SHARING'
    BAYES_OPT = 'BAYES_OPT'
    RANDOM = 'RANDOM'
    ENAS = 'ENAS'

class ParamsType(Enum):
    LOCAL_RECENT = 'LOCAL_RECENT'
    LOCAL_BEST = 'LOCAL_BEST'
    GLOBAL_RECENT = 'GLOBAL_RECENT'
    GLOBAL_BEST = 'GLOBAL_BEST'
    NONE = 'NONE'

class Jsonable():
    @classmethod
    def from_jsonable(cls, jsonable) -> object:
        return cls(**jsonable)

    def to_jsonable(self) -> any:
        jsonable = self.__dict__

        # Convert all nested jsonables & enums
        for (name, value) in jsonable.items():
            if isinstance(value, Jsonable):
                jsonable[name] = value.to_jsonable()
            elif isinstance(value, Enum):
                jsonable[name] = value.value

        return jsonable

    def __str__(self):
        return str(self.to_jsonable())

class TrainWorker(Jsonable):
    def __init__(self, 
                worker_id: str, 
                gpus: int = 0): # No. of GPUs allocated to worker
        self.worker_id = worker_id
        self.gpus = gpus

class Proposal(Jsonable):
    def __init__(self, 
                knobs: Knobs = None, 
                params_type: ParamsType = ParamsType.NONE, # Parameters to use for this trial
                to_eval=True, # Whether the model should be evaluated
                to_cache_params=False, # Whether this trial's parameters should be cached
                to_save_params=True, # Whether this trial's parameters should be persisted
                meta: dict = None): # Extra metadata associated with proposal
        self.knobs = knobs
        self.params_type = ParamsType(params_type)
        self.to_eval = to_eval
        self.to_cache_params = to_cache_params
        self.to_save_params = to_save_params
        self.meta = meta or {}

class ProposalResult(Jsonable):
    def __init__(self, 
                proposal: Proposal, 
                score: float, # Score for the proposal
                worker_id: str): # ID of worker that ran the proposal
        self.proposal = proposal if isinstance(proposal, Proposal) else Proposal(**proposal)
        self.score = score
        self.worker_id = worker_id