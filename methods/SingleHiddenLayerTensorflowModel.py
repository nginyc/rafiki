import tensorflow as tf
from tensorflow import keras
import pandas as pd
import json
import os
import numpy as np

from .BaseMethod import BaseMethod


class SingleHiddenLayerTensorflowModel(BaseMethod):
    def __init__(self, hidden_layer_units=2):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        
        with self._graph.as_default():
            self._model = self._build_model(hidden_layer_units)

    def fit(self, X, y):
        y = keras.utils.to_categorical(y, num_classes=2)
        with self._graph.as_default():
            with self._sess.as_default():
                self._model.fit(X, y, epochs=100, batch_size=32)

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
    def Load(self, model_dir, model_id, hyperparameters):
        model = SingleHiddenLayerTensorflowModel(**hyperparameters)
        model_file_path = os.path.join(model_dir, str(model_id) + '.h5')

        with model._graph.as_default():
            with model._sess.as_default():
                model._model = keras.models.load_model(model_file_path)

        return model


    def _build_model(self, hidden_layer_units):
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            hidden_layer_units,
            activation='relu'
        ))
        model.add(keras.layers.Dense(2, activation='softmax'))
        model.compile(
            optimizer=tf.train.AdamOptimizer(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


if __name__ == '__main__':
    # Temporarily use Kaggle titanic data
    LABEL_COLUMN = 'Survived'
    train_df = pd.read_csv('./data/kaggle-titanic/train_final.csv')
    # print(train_df)
    X = train_df[train_df.columns.difference(
        [LABEL_COLUMN])].values.astype(float)
    y = train_df[LABEL_COLUMN].values.astype(float)
    # print(json.dumps(X))

    # data = np.random.random((1000, 32))
    # labels = (np.random.random((1000)) > 0.5).astype(int)
    model = SingleHiddenLayerTensorflowModel(10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.shape(y) == np.shape(predictions)
    print('Accuracy', np.sum(predictions == y) / len(y))
