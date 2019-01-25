import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import json
import os
import tempfile
import numpy as np
import base64

from rafiki.config import APP_MODE
from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, utils
from rafiki.constants import TaskType, ModelDependency

class TfFeedForward(BaseModel):
    '''
    Implements a fully-connected feed-forward neural network with variable hidden layers on Tensorflow 
    for image classification
    '''
    @staticmethod
    def get_knob_config():
        return {
            'max_epochs': FixedKnob(100),
            'hidden_layer_count': IntegerKnob(1, 8 if APP_MODE != 'DEV' else 2),
            'hidden_layer_units': IntegerKnob(2, 128),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
            'max_image_size': CategoricalKnob([16, 32, 48]),
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph, config=config)
        
    def train(self, dataset_uri):
        max_image_size = self._knobs['max_image_size']
        bs = self._knobs['batch_size']
        max_epochs = self._knobs['max_epochs']

        utils.logger.log('Available devices: {}'.format(str(device_lib.list_local_devices())))

        # Define plot for loss against epochs
        utils.logger.define_plot('Loss Over Epochs', ['loss', 'early_stop_val_loss'], x_axis='epoch')

        # Load dataset
        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=max_image_size, 
                                                            mode='RGB')
        num_classes = dataset.classes
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        (images, norm_mean, norm_std) = utils.dataset.normalize_images(images)
        
        with self._graph.as_default():
            with self._sess.as_default():
                self._model = self._build_model(num_classes, dataset.image_size)
                self._model.fit(
                    np.asarray(images), 
                    np.asarray(classes), 
                    verbose=0,
                    epochs=max_epochs,
                    validation_split=0.05,
                    batch_size=bs,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
                        tf.keras.callbacks.LambdaCallback(on_epoch_end=self._on_train_epoch_end)
                    ]
                )

                # Compute train accuracy
                (loss, accuracy) = self._model.evaluate(images, classes)

        utils.logger.log('Train loss: {}'.format(loss))
        utils.logger.log('Train accuracy: {}'.format(accuracy))

        self._train_params = {
            'image_size': dataset.image_size,
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }

    def evaluate(self, dataset_uri):
        max_image_size = self._knobs['max_image_size']
        norm_mean = self._train_params['norm_mean']
        norm_std = self._train_params['norm_std']

        dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=max_image_size, 
                                                            mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        (images, _, _) = utils.dataset.normalize_images(images, norm_mean, norm_std)
        with self._graph.as_default():
            with self._sess.as_default():
                (loss, accuracy) = self._model.evaluate(np.asarray(images), np.asarray(classes))

        utils.logger.log('Validation loss: {}'.format(loss))

        return accuracy

    def predict(self, queries):
        image_size = self._train_params['image_size']
        norm_mean = self._train_params['norm_mean']
        norm_std = self._train_params['norm_std']

        images = utils.dataset.transform_images(queries, image_size=image_size, mode='RGB')
        (images, _, _) = utils.dataset.normalize_images(images, norm_mean, norm_std)

        with self._graph.as_default():
            with self._sess.as_default():
                probs = self._model.predict(images)
                
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

        # Save pre-processing params
        params['train_params'] = self._train_params

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

        # Load training params
        self._train_params = params['train_params']

    def _on_train_epoch_end(self, epoch, logs):
        loss = logs['loss']
        early_stop_val_loss = logs['val_loss']
        utils.logger.log(loss=loss, early_stop_val_loss=early_stop_val_loss, epoch=epoch)

    def _build_model(self, num_classes, image_size):
        units = self._knobs['hidden_layer_units']
        layers = self._knobs['hidden_layer_count']
        lr = self._knobs['learning_rate']
         
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(image_size, image_size, 3)))
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
        val_dataset_uri='data/fashion_mnist_for_image_classification_val.zip',
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
