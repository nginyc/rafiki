import abc
import numpy as np

from rafiki.constants import AdvisorType

class InvalidAdvisorTypeError(Exception): pass
class UnsupportedKnobTypeError(Exception): pass

class BaseAdvisor(abc.ABC):
    '''
    Rafiki's base advisor class
    '''   
    @abc.abstractmethod
    def __init__(self, knob_config):
        raise NotImplementedError()

    @abc.abstractmethod
    def propose(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, knobs, score):
        raise NotImplementedError()

# Generalized Advisor class that wraps & hides implementation-specific advisor class
class Advisor():
    def __init__(self, knob_config, advisor_type):
        self._advisor = self._make_advisor(knob_config, advisor_type)
        self._knob_config = knob_config

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

    def _make_advisor(self, knob_config, advisor_type):
        if advisor_type == AdvisorType.SKOPT:
            from .types.skopt_advisor import SkoptAdvisor
            return SkoptAdvisor(knob_config)
        elif advisor_type == AdvisorType.BTB_GP:
            from .types.btb_gp_advisor import BtbGpAdvisor
            return BtbGpAdvisor(knob_config)
        elif advisor_type == AdvisorType.ENAS:
            from .types.enas_advisor import EnasAdvisor
            return EnasAdvisor(knob_config)
        elif advisor_type == AdvisorType.RANDOM:
            from .types.random_advisor import RandomAdvisor
            return RandomAdvisor(knob_config)
        else:
            raise InvalidAdvisorTypeError()

    def _simplify_value(self, value):
        # TODO: Support int64 & other non-serializable data formats
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            return float(value)

        return value
