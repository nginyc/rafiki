#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import tensorflow as tf
from tensorflow import keras
import json
import tempfile
import numpy as np
import base64
import argparse

from rafiki.model import utils, BaseModel, IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, PolicyKnob
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

class TfFeedForward(BaseModel):
    '''
    Implements a fully-connected feed-forward neural network with variable hidden layers on Tensorflow for IMAGE_CLASSIFICATION
    '''
    @staticmethod
    def get_knob_config():
        return {
            'max_epochs': FixedKnob(10),
            'hidden_layer_count': IntegerKnob(1, 2),
            'hidden_layer_units': IntegerKnob(2, 128),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
            'max_image_size': CategoricalKnob([16, 32, 48]),
            'quick_train': PolicyKnob('EARLY_STOP') # Whether early stopping would be used
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph, config=config)
        self._model = None
        self._train_params = None
        
    def train(self, dataset_path, **kwargs):
        max_image_size = self._knobs['max_image_size']
        bs = self._knobs['batch_size']
        max_epochs = self._knobs['max_epochs']
        quick_train = self._knobs['quick_train']

        # Define plot for loss against epochs
        utils.logger.define_plot('Loss Over Epochs', ['loss', 'early_stop_val_loss'], x_axis='epoch')

        # Load dataset
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=max_image_size, 
                                                            mode='RGB')
        num_classes = dataset.classes
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        (images, norm_mean, norm_std) = utils.dataset.normalize_images(images)

        # Setup callbacks, adding early stopping if quick train
        callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end=self._on_train_epoch_end)]
        if quick_train:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2))
        
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
                    callbacks=callbacks
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

    def evaluate(self, dataset_path):
        max_image_size = self._knobs['max_image_size']
        norm_mean = self._train_params['norm_mean']
        norm_std = self._train_params['norm_std']

        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=max_image_size, 
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

        # Add train params
        params['train_params'] = json.dumps(self._train_params)

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
        h5_model_base64 = params['h5_model_base64']

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)

            # Load model from temp file
            with self._graph.as_default():
                with self._sess.as_default():
                    self._model = keras.models.load_model(tmp.name)
        
        # Add train params
        self._train_params = json.loads(params['train_params'])

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/fashion_mnist_train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/fashion_mnist_val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/fashion_mnist_test.zip', help='Path to test dataset')
    parser.add_argument('--query_path', type=str, default='examples/data/image_classification/fashion_mnist_test_1.png', 
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    test_model_class(
        model_file_path=__file__,
        model_class='TfFeedForward',
        task='IMAGE_CLASSIFICATION',
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )
