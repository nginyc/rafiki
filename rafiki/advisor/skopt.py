from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer
from collections import OrderedDict

from rafiki.model import CategoricalKnob, FixedKnob, IntegerKnob, FloatKnob

from .advisor import BaseKnobAdvisor, UnsupportedKnobTypeError

class SkoptKnobAdvisor(BaseKnobAdvisor):
    '''
    Uses `skopt`'s `Optimizer`
    '''   
    def start(self, total_trials, knob_config):
        self._knob_config = knob_config
        self._dimensions = self._get_dimensions(knob_config)
        n_random_starts = min(10, total_trials // 2) # Initially, no, of random starts
        self._optimizer = Optimizer(
            list(self._dimensions.values()),
            n_random_starts=n_random_starts,
            base_estimator='gp'
        )

    def propose(self):
        # Ask skopt
        point = self._optimizer.ask()
        knobs = { 
            name: value 
            for (name, value) 
            in zip(self._dimensions.keys(), point) 
        }
        
        return knobs

    def feedback(self, score, knobs):
        point = [ knobs[name] for name in self._dimensions.keys() ]
        self._optimizer.tell(point, -score)

    def _get_dimensions(self, knob_config):
        dimensions = OrderedDict({
            name: _knob_to_dimension(x)
                for (name, x)
                in knob_config.items()
        })
        return dimensions

def _knob_to_dimension(knob):
    if isinstance(knob, CategoricalKnob):
        return Categorical(knob.values)
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
    