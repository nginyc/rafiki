import tensorflow as tf
from tensorflow import keras
import json
import os
import tempfile
import numpy as np
import base64

from .dataset import load_dataset
from .BaseModel import BaseModel, InvalidModelParamsException

class SingleHiddenLayerTensorflowModel(BaseModel):

    def get_hyperparameter_config(self):
        return {
            'hyperparameters': {
                'hidden_layer_units': {
                    'type': 'int',
                    'range': [2, 128]
                },
                'epochs': {
                    'type': 'int',
                    'range': [1, 1]
                },
                'learning_rate': {
                    'type': 'float_exp',
                    'range': [1e-5, 1e-1]
                },
                'batch_size': {
                    'type': 'int_cat',
                    'values': [1, 2, 4, 8, 16, 32, 64, 128]
                }
            },
            'root_hyperparameters': ['hidden_layer_units', 'epochs', 'learning_rate', 'batch_size'],
            'conditional_hyperparameters': {}
        }

    def init(self, hyperparameters):
        self._batch_size = hyperparameters.get('batch_size')
        self._epochs = hyperparameters.get('epochs')
        self._hidden_layer_units = hyperparameters.get('hidden_layer_units')
        self._learning_rate = hyperparameters.get('learning_rate')

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        
        
    def train(self, dataset_uri):
        (images, labels) = self._load_dataset(dataset_uri)

        num_classes = len(np.unique(labels))

        X = images
        y = keras.utils.to_categorical(
            labels, 
            num_classes=num_classes
        )

        with self._graph.as_default():
            self._model = self._build_model(num_classes)
            with self._sess.as_default():
                self._model.fit(
                    X, 
                    y, 
                    epochs=self._epochs, 
                    batch_size=self._batch_size
                )

    def evaluate(self, dataset_uri):
        (images, labels) = self._load_dataset(dataset_uri)

        num_classes = len(np.unique(labels))

        X = images
        y = keras.utils.to_categorical(
            labels, 
            num_classes=num_classes
        )

        preds = self.predict(X)

        accuracy = sum(labels == preds) / len(y)
        return accuracy


    def predict(self, queries):
        X = np.array(queries)
        with self._graph.as_default():
            with self._sess.as_default():
                probs = self._model.predict(X)
                preds = np.argmax(probs, axis=1)

        return preds

    def destroy(self):
        self._sess.close()

    def dump_parameters(self):
        # TODO: Not save to & read from a file 

        # Save whole model to temp h5 file
        tmp = tempfile.NamedTemporaryFile(delete=False)
        with self._graph.as_default():
            with self._sess.as_default():
                self._model.save(tmp.name)
        
        # Read from temp h5 file & encode it to base64 string
        with open(tmp.name, 'rb') as f:
            h5_model_bytes = f.read()

        h5_model_base64 = base64.b64encode(h5_model_bytes).decode('utf-8')

        # Remove temp file
        os.remove(tmp.name)

        return {
            'h5_model_base64': h5_model_base64
        }

    def load_parameters(self, params):
        h5_model_base64 = params.get('h5_model_base64', None)

        if h5_model_base64 is None:
            raise InvalidModelParamsException()

        # TODO: Not save to & read from a file 

        # Convert back to bytes & write to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False)
        h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
        with open(tmp.name, 'wb') as f:
            f.write(h5_model_bytes)

        # Load model from temp file
        with self._graph.as_default():
            with self._sess.as_default():
                self._model = keras.models.load_model(tmp.name)
        
        # Remove temp file
        os.remove(tmp.name)


    def _load_dataset(self, dataset_uri):
        # Here, we use Rafiki's in-built dataset loader
        return load_dataset(dataset_uri) 

    def _build_model(self, num_classes):
        hidden_layer_units = self._hidden_layer_units
        learning_rate = self._learning_rate

        model = keras.Sequential()
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(
            hidden_layer_units,
            activation=tf.nn.relu
        ))
        model.add(keras.layers.Dense(
            num_classes, 
            activation=tf.nn.softmax
        ))
        
        model.compile(
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

