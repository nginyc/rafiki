from btb.tuning import GP
from btb import HyperParameter, ParamTypes

from .advisor import BaseAdvisor

class BtbGpAdvisor(BaseAdvisor):
    '''
    Uses BTB's GP tuner
    '''   
    def __init__(self, knob_config):
        # TODO: Support conditional knobs
        knobs = knob_config['knobs']
        tunables = self._get_tunables(knobs)

        # TODO: Allow configuration of tuner
        self._tuner = GP(tunables=tunables)

    def propose(self):
        knobs = self._tuner.propose()
        return knobs

    def feedback(self, knobs, score):
        self._tuner.add(knobs, score)

    def _get_tunables(self, knobs):
        tunables = [
            _knob_to_tunable(name, knob_config)
                for (name, knob_config)
                in knobs.items()
        ]
        return tunables

_KNOB_TYPE_TO_TUNABLE_TYPE = {
    'int': ParamTypes.INT,
    'int_exp': ParamTypes.INT_EXP,
    'int_cat': ParamTypes.INT_CAT,
    'float': ParamTypes.FLOAT,
    'float_exp': ParamTypes.FLOAT_EXP,
    'float_cat': ParamTypes.FLOAT_CAT,
    'string': ParamTypes.STRING,
    'bool': ParamTypes.BOOL
}

_KNOB_CONFIG_TO_TUNABLE_RANGE = {
    ParamTypes.INT: (lambda x: x['range']),
    ParamTypes.INT_EXP: (lambda x: x['range']),
    ParamTypes.INT_CAT: (lambda x: x['values']),
    ParamTypes.FLOAT: (lambda x: x['range']),
    ParamTypes.FLOAT_EXP: (lambda x: x['range']),
    ParamTypes.FLOAT_CAT: (lambda x: x['values']),
    ParamTypes.STRING: (lambda x: x['values']),
    ParamTypes.BOOL: (lambda x: x['values'])
}

def _knob_to_tunable(name, knob_config):
    tunable_type = _KNOB_TYPE_TO_TUNABLE_TYPE[knob_config['type']]
    tunable_range = _KNOB_CONFIG_TO_TUNABLE_RANGE[tunable_type](knob_config)
    return (name, HyperParameter(tunable_type, tunable_range))