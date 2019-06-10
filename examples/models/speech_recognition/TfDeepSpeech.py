import tensorflow as tf

from rafiki.model import BaseModel, FixedKnob, FloatKnob, CategoricalKnob
# InvalidModelParamsException, test_model_class, IntegerKnob, dataset_utils, logger

FLAGS = tf.app.flags.FLAGS


class TfDeepSpeech(BaseModel):
    '''
    Implements a speech recognition neural network model developed by Baidu. It contains five hiddlen layers.
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(3),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self._graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._graph, config=config)

    def train(self, dataset_uri):
        pass

    def evaluate(self, dataset_uri):
        pass

    def predict(self, queries):
        pass

    def destroy(self):
        pass

    def dump_parameters(self):
        pass

    def load_parameters(self, params):
        pass


if __name__ == '__main__':
    pass
