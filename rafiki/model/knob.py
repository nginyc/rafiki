import abc
import json
from typing import Union, List
from enum import Enum

class BaseKnob(abc.ABC):
    '''
    The base class for a knob type.
    '''

    # Data type of a realized value of this knob
    @property
    def value_type(self) -> type:
        raise NotImplementedError()

class KnobValue():
    '''
    Wrapper for a discrete value of type ``int``, ``float``, ``bool``, ``str``, ``list``.
    '''
    def __init__(self, value: any):
        (self._value, self._dtype) = self._parse_value(value)

    @property
    def value(self):
        return self._value

    @property
    def dtype(self) -> type:
        return self._dtype

    @staticmethod
    def _parse_value(value):
        value_type = None

        if isinstance(value, int):
            value_type = int
        elif isinstance(value, float):
            value_type = float
        elif isinstance(value, bool):
            value_type = bool
        elif isinstance(value, str):
            value_type = str
        elif isinstance(value, list):
            value_type = list
            value = [KnobValue(x) for x in value]
        else:
            raise TypeError('Only the following knob value data types are supported: `int`, `float`, `bool`, `str`, `list`')
        
        return (value, value_type)

class CategoricalKnob(BaseKnob):
    '''
    Knob type representing a variable discrete value.
    ``values`` is a list of candidate values for this knob type; a realization of this knob type would be an element of ``values``.
    Elements of ``values`` must be of the same type. 
    If the same ``KnobValue`` instance is reused at different indices in ``values``, they are considered to be semantically identical for the purposes of search.  
    '''
    def __init__(self, values):
        (self._values, self._value_type) = self._validate_values(values)

    @property
    def value_type(self):
        return self._value_type

    @property
    def values(self) -> list:
        return self._values

    @staticmethod
    def _validate_values(values):
        values = [KnobValue(x) if not isinstance(x, KnobValue) else x for x in values]

        if len(values) == 0:
            raise ValueError('Length of `values` should at least 1')

        if any([values[0].dtype is not x.dtype for x in values]):
            raise TypeError('`values` should have elements of the same type')

        value_type = values[0].dtype

        return (values, value_type)

class FixedKnob(BaseKnob):
    '''
    Knob type representing a fixed discrete value.
    Essentially, this represents a knob type that does not require tuning.
    '''
    def __init__(self, value):
        self._value = KnobValue(value) if not isinstance(value, KnobValue) else value
        self._value_type = self._value.dtype

    @property
    def value_type(self):
        return self._value_type

    @property
    def value(self):
        return self._value

POLICIES = ['QUICK_TRAIN', 'SKIP_TRAIN', 'QUICK_EVAL', 'DOWNSCALE']

class PolicyKnob(BaseKnob):
    '''
    Knob type representing whether a certain policy should be activated, as a boolean.
    E.g. the `QUICK_TRAIN` policy knob decides whether the model should stop model training early, or not. 
    Offering the ability to activate different policies can optimize hyperparameter search for your model. 
    Activation of all policies default to false.

    =====================       =====================
    **Policy**                  Description
    ---------------------       ---------------------        
    ``QUICK_TRAIN``             Whether model should stop training early in `train()`, e.g. with use of early stopping or reduced no. of epochs
    ``SKIP_TRAIN``              Whether model should skip training its parameters
    ``QUICK_EVAL``              Whether model should stop evaluation early in `evaluate()`, e.g. by evaluating on only a subset of the validation dataset
    ``DOWNSCALE``               Whether a smaller version of the model should be constructed e.g. with fewer layers
    =====================       =====================
    
    '''
    def __init__(self, policy):
        if policy not in POLICIES:
            raise ValueError('Policy type must be one of {}'.format(POLICIES))
        self._policy = policy
    
    @property
    def value_type(self):
        return bool

    @property
    def policy(self):
        return self._policy


class IntegerKnob(BaseKnob):
    '''
    Knob type representing `any` ``int`` value within a specific interval [``value_min``, ``value_max``].
    ``is_exp`` specifies whether the knob value should be scaled exponentially.
    '''

    def __init__(self, value_min, value_max, is_exp=False):
        self._validate_values(value_min, value_max)
        self._value_min = value_min
        self._value_max = value_max
        self._is_exp = is_exp

    @property
    def value_type(self):
        return int
    
    @property
    def value_min(self) -> int:
        return self._value_min

    @property
    def value_max(self) -> int:
        return self._value_max
    
    @property
    def is_exp(self) -> bool:
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
    Knob type representing a ``float`` value within a specific interval [``value_min``, ``value_max``].
    ``is_exp`` specifies whether the knob value should be scaled exponentially.
    '''

    def __init__(self, value_min, value_max, is_exp=False):
        self._validate_values(value_min, value_max)
        self._value_min = float(value_min)
        self._value_max = float(value_max)
        self._is_exp = is_exp

    @property
    def value_type(self):
        return float
    
    @property
    def value_min(self) -> float:
        return self._value_min

    @property
    def value_max(self) -> float:
        return self._value_max
    
    @property
    def is_exp(self) -> bool:
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
    Knob type representing a fixed-size list of knobs. 
    ``items`` is a list of ``BaseKnob``; a realization of this knob type would be a list of values, each value corresponding to the realization of each element in ``items``.
    It is assumed that the realization of a element in the list can depend on the realization of elements preceding it. 
    '''

    def __init__(self, items):
        (self._list_len, self._items) = self._validate_values(items)

    @property
    def value_type(self):
        return list

    @property  
    def items(self) -> List[BaseKnob]:
        return self._items

    def __len__(self):
        return self._list_len

    @staticmethod
    def _validate_values(items):
        for (i, knob) in enumerate(items):
            if not isinstance(knob, BaseKnob):
                raise ValueError('Item {} should be of type `BaseKnob`'.format(i))

        list_len = len(items)
        return (list_len, items)
