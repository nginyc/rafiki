from btb.tuning import GP
from btb import HyperParameter, ParamTypes
import numpy as np

from .BaseModel import BaseModel
from .model import InvalidModelException

def create_tuner(hyperparameters_config):
    hyperparameters = hyperparameters_config['hyperparameters']
    tunables = _get_tunables(hyperparameters)
    
    # TODO: Allow configuration of tuner
    tuner = GP(tunables=tunables)
    return tuner

def propose_with_tuner(tuner):
    hyperparameters = tuner.propose()
    
    # Simplify hyperparameters to use JSON serializable values
    # TODO: Support int64 & other non-serializable data formats
    hyperparameters = {
        name: _simplify_value(value)
            for name, value
            in hyperparameters.items()
    }

    return hyperparameters

def train_tuner(tuner, hyperparameters_list, scores):
    if len(hyperparameters_list) == 0:
        return tuner
    
    tuner.add(hyperparameters_list, scores)
    return tuner

def _get_tunables(hyperparameters):
    # TODO: Support conditional hyperparameters
    tunables = [
        _hyperparameter_to_tunable(name, hyperparameter_config)
            for (name, hyperparameter_config)
            in hyperparameters.items()
    ]
    return tunables

def _simplify_value(value):
    if isinstance(value, np.int64):
        return int(value)

    return value
    

_HYPERPARAMETER_TYPE_TO_TUNABLE_TYPE = {
    'int': ParamTypes.INT,
    'int_exp': ParamTypes.INT_EXP,
    'int_cat': ParamTypes.INT_CAT,
    'float': ParamTypes.FLOAT,
    'float_exp': ParamTypes.FLOAT_EXP,
    'float_cat': ParamTypes.FLOAT_CAT,
    'string': ParamTypes.STRING,
    'bool': ParamTypes.BOOL
}

_HYPERPARAMETER_CONFIG_TO_TUNABLE_RANGE = {
    ParamTypes.INT: (lambda x: x['range']),
    ParamTypes.INT_EXP: (lambda x: x['range']),
    ParamTypes.INT_CAT: (lambda x: x['values']),
    ParamTypes.FLOAT: (lambda x: x['range']),
    ParamTypes.FLOAT_EXP: (lambda x: x['range']),
    ParamTypes.FLOAT_CAT: (lambda x: x['values']),
    ParamTypes.STRING: (lambda x: x['values']),
    ParamTypes.BOOL: (lambda x: x['values'])
}

def _hyperparameter_to_tunable(name, hyperparameter_config):
    tunable_type = _HYPERPARAMETER_TYPE_TO_TUNABLE_TYPE[hyperparameter_config['type']]
    tunable_range = _HYPERPARAMETER_CONFIG_TO_TUNABLE_RANGE[tunable_type](hyperparameter_config)
    return (name, HyperParameter(tunable_type, tunable_range))