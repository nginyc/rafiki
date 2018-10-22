import abc

from rafiki.constants import AdvisorType

class InvalidAdvisorTypeException(Exception): pass

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


def make_advisor(knob_config, advisor_type=AdvisorType.BTB_GP):
    if advisor_type == AdvisorType.BTB_GP:
        from .btb_gp_advisor import BtbGpAdvisor
        return BtbGpAdvisor(knob_config)
    else:
        raise InvalidAdvisorTypeException()