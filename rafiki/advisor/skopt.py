from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer
    
from .advisor import BaseKnobAdvisor, UnsupportedKnobTypeError
from .knob import CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob

class SkoptKnobAdvisor(BaseKnobAdvisor):
    '''
    Uses `skopt`'s `Optimizer`
    '''   
    def start(self, knob_config):
        self._dimensions = self._get_dimensions(knob_config)
        self._optimizer = Optimizer(list(self._dimensions.values()))

    def propose(self):
        point = self._optimizer.ask()
        return { knob : value for (knob, value) in zip(self._dimensions.keys(), point) }

    def feedback(self, knobs, score):
        point = [ knobs[name] for name in self._dimensions.keys() ]
        self._optimizer.tell(point, -score)

    def _get_dimensions(self, knob_config):
        dimensions = {
            name: _knob_to_dimension(x)
                for (name, x)
                in knob_config.items()
        }
        return dimensions

def _knob_to_dimension(knob):
    if isinstance(knob, CategoricalKnob):
        return Categorical(knob.values)
    elif isinstance(knob, FixedKnob):
        return Categorical([knob.value])
    elif isinstance(knob, IntegerKnob):
        return Integer(knob.value_min, knob.value_max)
    elif isinstance(knob, FloatKnob):
        if knob.is_exp:
            # Avoid error in skopt when low/high are 0
            value_min = knob.value_min if knob.value_min != 0 else 1e-12
            value_max = knob.value_max if knob.value_max != 0 else 1e-12
            return Real(value_min, value_max, 'log-uniform')
        else:
            return Real(knob.value_min, knob.value_max, 'uniform')
    else:
        raise UnsupportedKnobTypeError(knob.__class__)


def _unzeroify(self, value):
    if value == 0:
        return 1e-9
    