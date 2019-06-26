import random

from rafiki.model import BaseModel

class Model(BaseModel):
    '''
    Model that errors while loading params
    '''
    @staticmethod
    def get_knob_config():
        return {}
       
    def train(self, dataset_path, **kwargs):
        pass

    def evaluate(self, dataset_path):
        return random.random()

    def predict(self, queries):
        return [1 for x in queries]

    def dump_parameters(self):
        return {}

    def load_parameters(self, params):
        raise Exception()
