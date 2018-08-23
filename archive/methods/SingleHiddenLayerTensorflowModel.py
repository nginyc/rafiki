import tensorflow as tf
from tensorflow import keras
import pandas as pd
import json
import os
import numpy as np

from .BaseMethod import BaseMethod


class SingleHiddenLayerTensorflowModel(BaseMethod):
    def __init__(self, num_classes, hidden_layer_units=2, \
                epochs=10, learning_rate=0.001, batch_size=32):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._batch_size = batch_size
        self._epochs = epochs
        self._num_classes = num_classes
        
        with self._graph.as_default():
            self._model = self._build_model(num_classes, hidden_layer_units, learning_rate)

    def fit(self, X, y):
        y = keras.utils.to_categorical(
            y, 
            num_classes=self._num_classes
        )
        with self._graph.as_default():
            with self._sess.as_default():
                self._model.fit(X, y, epochs=self._epochs, 
                                batch_size=self._batch_size)

    def predict(self, X):
        predictions = self.predict_proba(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        with self._graph.as_default():
            with self._sess.as_default():
                predictions = self._model.predict(X)

        return predictions

    def destroy(self):
        self._sess.close()

    @classmethod
    def Save(self, model, model_dir, model_id):
        model_file_path = os.path.join(model_dir, str(model_id) + '.h5')

        with model._graph.as_default():
            with model._sess.as_default():
                model._model.save(model_file_path)

        return model_file_path
        
    @classmethod
    def Load(self, model_dir, model_id, num_classes, hyperparameters):
        model = SingleHiddenLayerTensorflowModel(num_classes, **hyperparameters)
        model_file_path = os.path.join(model_dir, str(model_id) + '.h5')

        with model._graph.as_default():
            with model._sess.as_default():
                model._model = keras.models.load_model(model_file_path)

        return model


    def _build_model(self, num_classes, hidden_layer_units, learning_rate):
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
            optimizer=tf.train.AdamOptimizer(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

