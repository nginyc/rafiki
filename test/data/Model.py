import random
import numpy as np

from rafiki.model import BaseModel, IntegerKnob, FixedKnob, CategoricalKnob, FloatKnob, PolicyKnob

class Model(BaseModel):
    '''
    A mock model
    '''
    @staticmethod
    def get_knob_config():
        return {
            'int': IntegerKnob(1, 32),
            'float': FloatKnob(1e-5, 1),
            'cat': CategoricalKnob(['a', 'b', 'c']),
            'fixed': FixedKnob('fixed'),
            'policy': PolicyKnob('EARLY_STOP')
        }

    def train(self, dataset_path, **kwargs):
        pass

    def evaluate(self, dataset_path):
        return random.random()

    def predict(self, queries):
        return [1 for x in queries]

    def dump_parameters(self):
        return {'int': 100, 'str': 'str', 'float': 0.001, 'np': np.array([1, 2, 3])}

    def load_parameters(self, params):
        pass
