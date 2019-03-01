import os
import numpy as np
from typing import Dict

class ParamStore(object):
    '''
    Store API that retrieves and stores parameters.
    '''
    def __init__(self):
        # Stores parameters in-memory
        self._params = {} 
    
    '''
    Retrieves parameters from underlying storage.

    :param str trial_id: Unique trial ID for parameters
    :param dict params: Parameters as a light { <name>: <id> } dictionary
    :returns: Parameters as a heavy { <name>: <numpy array> } dictionary
    :rtype: dict
    '''
    def retrieve_params(self, trial_id: str, params: Dict[str, str]):
        out_params = {}
        for (name, param_id) in params:
            if param_id in self._params:
                out_params[name] = self._params[param_id]

        return out_params

    '''
    Stores parameters into underlying storage.

    :param str trial_id: Unique trial ID for parameters
    :param dict params: Parameters as a heavy { <name>: <numpy array> } dictionary
    :returns: Parameters as a light { <name>: <id> } dictionary
    :rtype: dict
    '''
    def store_params(self, trial_id: str, params: Dict[str, np.array]):
        out_params = {}
        for (name, value) in params.items():
            param_id = 'trial_{}_param_{}'.format(trial_id, name)
            self._params[param_id] = value
            out_params[name] = param_id

        return out_params
        

