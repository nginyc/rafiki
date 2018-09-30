import abc

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
    def set_result_of_proposal(self, knobs, score):
        raise NotImplementedError()

    