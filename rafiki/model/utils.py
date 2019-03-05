from .dataset import DatasetUtils
from .log import LoggerUtils

class ModelUtils():
    def __init__(self):
        self._trial_id = None
        self.dataset = DatasetUtils()
        self.logger = LoggerUtils()

# Initialize a global instance
utils = ModelUtils()
logger = utils.logger
dataset = utils.dataset
    
