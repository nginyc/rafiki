import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import json
import os
import tempfile
import numpy as np
import base64

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, dataset_utils, logger
from rafiki.constants import TaskType, ModelDependency

class TfFeedForward(BaseModel):
    '''
    Implements a fully-connected feed-forward neural network with variable hidden layers on Tensorflow 
    for simple image classification
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(3),
            'hidden_layer_count': IntegerKnob(1, 2),
            'hidden_layer_units': IntegerKnob(2, 128),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
            'image_size': FixedKnob(32)
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self._graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._graph, config=config)
        
    def train(self, dataset_uri):
        im_sz = self._knobs.get('image_size')
        bs = self._knobs.get('batch_size')
        ep = self._knobs.get('epochs')

        logger.log('Available devices: {}'.format(str(device_lib.list_local_devices())))

        # Define 2 plots: Loss against time, loss against epochs
        logger.define_loss_plot()
        logger.define_plot('Loss Over Time', ['loss'])

        dataset = dataset_utils.load_dataset_of_image_files(dataset_uri, image_size=[im_sz, im_sz])
        num_classes = dataset.classes
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        images = np.asarray(images)
        classes = np.asarray(classes)

        with self._graph.as_default():
            self._model = self._build_model(num_classes)
            with self._sess.as_default():
                self._model.fit(
                    images, 
                    classes, 
                    verbose=0,
                    epochs=ep,
                    batch_size=bs,
                    callbacks=[
                        tf.keras.callbacks.LambdaCallback(on_epoch_end=self._on_train_epoch_end)
                    ]
                )

                # Compute train accuracy
                (loss, accuracy) = self._model.evaluate(images, classes)
                logger.log('Train loss: {}'.format(loss))
                logger.log('Train accuracy: {}'.format(accuracy))

    def evaluate(self, dataset_uri):
        im_sz = self._knobs.get('image_size')

        dataset = dataset_utils.load_dataset_of_image_files(dataset_uri, image_size=[im_sz, im_sz])
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        images = np.asarray(images)
        classes = np.asarray(classes)

        with self._graph.as_default():
            with self._sess.as_default():
                (loss, accuracy) = self._model.evaluate(images, classes)
                logger.log('Test loss: {}'.format(loss))

        return accuracy

    def predict(self, queries):
        im_sz = self._knobs.get('image_size')

        X = dataset_utils.resize_as_images(queries, image_size=[im_sz, im_sz])
        with self._graph.as_default():
            with self._sess.as_default():
                probs = self._model.predict(X)
                
        return probs.tolist()

    def destroy(self):
        self._sess.close()

    def dump_parameters(self):
        params = {}

        # Save model parameters
        with tempfile.NamedTemporaryFile() as tmp:
            # Save whole model to temp h5 file
            with self._graph.as_default():
                with self._sess.as_default():
                    self._model.save(tmp.name)
        
            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                h5_model_bytes = f.read()

            params['h5_model_base64'] = base64.b64encode(h5_model_bytes).decode('utf-8')

        return params

    def load_parameters(self, params):
        # Load model parameters
        h5_model_base64 = params.get('h5_model_base64', None)
        if h5_model_base64 is None:
            raise InvalidModelParamsException()

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)

            # Load model from temp file
            with self._graph.as_default():
                with self._sess.as_default():
                    self._model = keras.models.load_model(tmp.name)

    def _on_train_epoch_end(self, epoch, logs):
        loss = logs['loss']
        logger.log_loss(loss, epoch)

    def _build_model(self, num_classes):
        units = self._knobs.get('hidden_layer_units')
        layers = self._knobs.get('hidden_layer_count')
        lr = self._knobs.get('learning_rate')
        im_sz = self._knobs.get('image_size')

        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(im_sz, im_sz,)))
        model.add(keras.layers.BatchNormalization())

        for _ in range(layers):
            model.add(keras.layers.Dense(units, activation=tf.nn.relu))

        model.add(keras.layers.Dense(
            num_classes, 
            activation=tf.nn.softmax
        ))
        
        model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='TfFeedForward',
        task=TaskType.IMAGE_CLASSIFICATION,
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0'
        },
        train_dataset_uri='data/fashion_mnist_for_image_classification_train.zip',
        test_dataset_uri='data/fashion_mnist_for_image_classification_test.zip',
        queries=[
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], 
            [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], 
            [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], 
            [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], 
            [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], 
            [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], 
            [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], 
            [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], 
            [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ]
    )
