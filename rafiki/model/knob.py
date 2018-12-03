import abc
import json

class BaseKnob(abc.ABC):
    '''
    The base class for a knob type.
    '''

    # TODO: Support conditional and validation logic

    def __init__(self, knob_args={}):
        self._knob_args = knob_args

    def to_json(self):
        return json.dumps({
            'type': self.__class__.__name__,
            'args': self._knob_args
        })

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)

        if 'type' not in json_dict or 'args' not in json_dict:
            raise ValueError('Invalid JSON representation of knob: {}.'.format(json_str))

        knob_type = json_dict['type']
        knob_args = json_dict['args']
        knob_classes = [CategoricalKnob, IntegerKnob, FloatKnob]
        for clazz in knob_classes:
            if clazz.__name__ == knob_type:
                return clazz(**knob_args)

        raise ValueError('Invalid knob type: {}'.format(knob_type))

class CategoricalKnob(BaseKnob):
    '''
    Knob type representing a categorical value of type ``int``, ``float``, ``bool`` or ``str``.
    A generated value of this knob would be an element of ``values``.
    '''
    def __init__(self, values):
        knob_args = { 'values': values }
        super().__init__(knob_args)
        self._values = values
        (self._value_type) = self._validate_values(values)

    @property
    def value_type(self):
        return self._value_type

    @property
    def values(self):
        return self._values

    @staticmethod
    def _validate_values(values):
        if len(values) == 0:
            raise ValueError('Length of `values` should at least 1')
    
        if isinstance(values[0], int):
            value_type = int
        elif isinstance(values[0], float):
            value_type = float
        elif isinstance(values[0], bool):
            value_type = bool
        elif isinstance(values[0], str):
            value_type = str
        else:
            raise TypeError('Only the following types for `values` are supported: `int`, `float`, `bool`, `str`')
        
        if any([not isinstance(x, value_type) for x in values]):
            raise TypeError('`values` should have elements of the same type')

        return (value_type)

class IntegerKnob(BaseKnob):
    '''
    Knob type epresenting `any` ``int`` value within a specific interval [``value_min``, ``value_max``].
    ``is_exp`` specifies whether the knob value should be scaled exponentially.
    '''

    def __init__(self, value_min, value_max, is_exp=False):
        knob_args = { 'value_min': value_min, 'value_max': value_max, 'is_exp': is_exp }
        super().__init__(knob_args)
        self._validate_values(value_min, value_max)
        self._value_min = value_min
        self._value_max = value_max
        self._is_exp = is_exp
    
    @property
    def value_min(self):
        return self._value_min

    @property
    def value_max(self):
        return self._value_max
    
    @property
    def is_exp(self):
        return self._is_exp

    @staticmethod
    def _validate_values(value_min, value_max):
        if not isinstance(value_min, int):
            raise ValueError('`value_min` should be an `int`')
        
        if not isinstance(value_max, int):
            raise ValueError('`value_max` should be an `int`')

        if value_min > value_max:
            raise ValueError('`value_max` should be at least `value_min`')
        

class FloatKnob(BaseKnob):
    '''
    Knob type representing `any` ``float`` value within a specific interval [``value_min``, ``value_max``].
    ``is_exp`` specifies whether the knob value should be scaled exponentially.
    '''

    def __init__(self, value_min, value_max, is_exp=False):
        knob_args = { 'value_min': value_min, 'value_max': value_max, 'is_exp': is_exp }
        super().__init__(knob_args)
        self._validate_values(value_min, value_max)
        self._value_min = value_min
        self._value_max = value_max
        self._is_exp = is_exp
    
    @property
    def value_min(self):
        return self._value_min

    @property
    def value_max(self):
        return self._value_max
    
    @property
    def is_exp(self):
        return self._is_exp

    @staticmethod
    def _validate_values(value_min, value_max):
        if not isinstance(value_min, float) and not isinstance(value_min, int):
            raise ValueError('`value_min` should be a `float` or `int`')
        
        if not isinstance(value_max, float) and not isinstance(value_max, int):
            raise ValueError('`value_max` should be a `float` or `int`')

        if value_min > value_max:
            raise ValueError('`value_max` should be at least `value_min`')


def deserialize_knob_config(knob_config_str):
    knob_config = {
        name: BaseKnob.from_json(knob_str)
        for (name, knob_str) in json.loads(knob_config_str).items()
    }
    return knob_config

def serialize_knob_config(knob_config):
    knob_config_str = json.dumps({
        name: knob.to_json()
        for (name, knob) in knob_config.items()
    })
    return knob_config_str
    