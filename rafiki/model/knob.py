import abc
import json
import pickle

class BaseKnob(abc.ABC):
    '''
    The base class for a knob type.
    '''

    # TODO: Support conditional and validation logic

    def __init__(self):
        pass

class CategoricalKnob(BaseKnob):
    '''
    Knob type representing a categorical value of type ``int``, ``float``, ``bool`` or ``str``.
    A generated value of this knob would be an element of ``values``.
    '''
    def __init__(self, values):
        super().__init__()
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

class FixedKnob(BaseKnob):
    '''
    Knob type representing a single fixed value of type ``int``, ``float``, ``bool`` or ``str``.
    Essentially, this represents a knob that does not require tuning.
    '''
    def __init__(self, value):
        super().__init__()
        self._value = value
        (self._value_type) = self._validate_value(value)

    @property
    def value_type(self):
        return self._value_type

    @property
    def value(self):
        return self._value

    @staticmethod
    def _validate_value(value):
        if isinstance(value, int):
            value_type = int
        elif isinstance(value, float):
            value_type = float
        elif isinstance(value, bool):
            value_type = bool
        elif isinstance(value, str):
            value_type = str
        else:
            raise TypeError('Only the following types for `value` are supported: `int`, `float`, `bool`, `str`')
        
        return (value_type)

class IntegerKnob(BaseKnob):
    '''
    Knob type epresenting `any` ``int`` value within a specific interval [``value_min``, ``value_max``].
    ``is_exp`` specifies whether the knob value should be scaled exponentially.
    '''

    def __init__(self, value_min, value_max, is_exp=False):
        super().__init__()
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
        super().__init__()
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

class ListKnob(BaseKnob):
    '''
    Knob type represent a list of knobs of a static size 
    '''
    def __init__(self, list_len, get_item=None, items=None):
        (self._list_len, self._items) = \
            self._validate_values(list_len, get_item, items)
        super().__init__()

    @property  
    def items(self):
        return self._items

    @property    
    def list_len(self):
        return self._list_len

    @staticmethod
    def _validate_values(list_len, get_item, items):
        if not isinstance(list_len, int) or list_len < 0:
            raise ValueError('`len_mlist_lenin` should be a non-negative `int`')
        
        if items is None:
            if get_item is None:
                raise ValueError('`get_item` should be specified if `items` is not')

            items = [get_item(i) for i in range(list_len)]

        for (i, knob) in enumerate(items):
            if not isinstance(knob, BaseKnob):
                raise ValueError('Item {} should be of type `BaseKnob`'.format(i))

        return (list_len, items)

class DynamicListKnob(BaseKnob):
    '''
    Knob type represent a list of knobs of a dynamic size 
    '''

    def __init__(self, len_min, len_max, get_item=None, items=None):
        (self._len_min, self._len_max, self._items) = \
            self._validate_values(len_min, len_max, get_item, items)
        super().__init__()
        
    @property
    def items(self):
        return self._items

    @property
    def len_min(self):
        return self._len_min
    
    @property
    def len_max(self):
        return self._len_max
    
    @staticmethod
    def _validate_values(len_min, len_max, get_item, items):
        if not isinstance(len_min, int) or len_min < 0:
            raise ValueError('`len_min` should be a non-negative `int`')
        
        if not isinstance(len_max, int) or len_max < len_min:
            raise ValueError('`len_max` should be a `int` at least `len_min`')

        if items is None:
            if get_item is None:
                raise ValueError('`get_item` should be specified if `items` is not')

            items = [get_item(i) for i in range(0, len_max)]

        for (i, knob) in enumerate(items):
            if not isinstance(knob, BaseKnob):
                raise ValueError('Item {} should be of type `BaseKnob`'.format(i))

        return (len_min, len_max, items)

def deserialize_knob_config(knob_config_bytes):
    knob_config = pickle.loads(knob_config_bytes)
    return knob_config

def serialize_knob_config(knob_config):
    knob_config_bytes = pickle.dumps(knob_config)
    return knob_config_bytes
    