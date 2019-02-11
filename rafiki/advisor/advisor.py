import abc
import numpy as np

class UnsupportedKnobTypeError(Exception): pass

class BaseAdvisor(abc.ABC):
    '''
    Rafiki's base advisor class.
    '''   
    @abc.abstractmethod
    def start(self, knob_config: dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def propose(self) -> (dict, str):
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, score: float, knobs: dict, param_id: str = None):
        raise NotImplementedError()

# Generalized Advisor class that wraps & hides implementation-specific advisor class
class Advisor():
    def __init__(self, knob_config, advisor=None):
        # Default advisor as `SkoptAdvisor`
        if advisor is None:
            from .types.skopt_advisor import SkoptAdvisor
            advisor = SkoptAdvisor() 

        self._advisor = advisor
        self._knob_config = knob_config
        advisor.start(knob_config)

    @property
    def knob_config(self):
        return self._knob_config

    def propose(self):
        knobs = self._advisor.propose()

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        return knobs

    def feedback(self, knobs, score):
        self._advisor.feedback(knobs, score)

    def _simplify_value(self, value):
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            return float(value)

        return value
