from .dataset import DatasetUtils
from .log import LoggerUtils

class ModelUtils():
    def __init__(self):
        self._trial_id = None
        self.dataset = DatasetUtils()
        self.logger = LoggerUtils()

    '''
    Gets the current trial ID, which will be unique across trials.
    Useful for scoping/namespacing (e.g. variable scopes in TensorFlow)
    '''
    @property
    def trial_id(self):
        return self._trial_id

    # - INTERNAL METHOD -
    # Sets the current trial ID.
    # Before a trial, this method will be called by Rafiki to set this.
    def set_trial_id(self, trial_id):
        self._trial_id = trial_id


# Initialize a global instance
utils = ModelUtils()
logger = utils.logger
dataset = utils.dataset
    
