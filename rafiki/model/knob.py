import abc

# TODO: Add documentation for each knob

class BaseKnob(abc.ABC):
    # TODO: Support conditional and validation logic
    pass

class CategoricalKnob(BaseKnob):
    '''
    Knob representing a categorical value of type `int`, `float`, `bool` or `str`.
    A generated value of this knob must be an element of `values`.
    '''

    def __init__(self, values):
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
    Knob representing any `int` value within a specific interval [`value_min`, `value_max`].
    `is_exp` specifies whether the knob value should be scaled exponentially.
    '''

    def __init__(self, value_min, value_max, is_exp=False):
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
    Knob representing any `float` value within a specific interval [`value_min`, `value_max`].
    `is_exp` specifies whether the knob value should be scaled exponentially.
    '''

    def __init__(self, value_min, value_max, is_exp=False):
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
        