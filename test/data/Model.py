from rafiki.model import BaseModel, IntegerKnob, FixedKnob, CategoricalKnob, FloatKnob
import random

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
            'fixed': FixedKnob('fixed')
        }

    def __init__(self, **knobs):
        pass
       
    def train(self, dataset_uri):
        pass

    def evaluate(self, dataset_uri):
        return random.random()

    def predict(self, queries):
        return [1 for x in queries]

    def destroy(self):
        pass

    def dump_parameters(self):
        return { 'param1': 100, 'param2': None, 'param3': [1, 2, 3] }

    def load_parameters(self, params):
        pass
