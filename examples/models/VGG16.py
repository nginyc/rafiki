import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import json
import os
import tempfile
import numpy as np
import base64
import abc
from urllib.parse import urlparse, parse_qs 

from rafiki.dataset import load_dataset
from rafiki.model import BaseModel, InvalidModelParamsException

class VGG16(BaseModel):

    def get_knob_config(self):
        return {
            'knobs': {
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
            'root_knobs': ['hidden_layer_units', 'epochs', 'learning_rate', 'batch_size'],
            'conditional_knobs': {}
        }

    def init(self, knobs):
        self._batch_size = knobs.get('batch_size')
        self._epochs = knobs.get('epochs')
        self._learning_rate = knobs.get('learning_rate')

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

    def train(self, dataset_uri):
        (images, labels) = self._load_dataset(dataset_uri)
        images = images.reshape(-1, 784)
        images = np.dstack([images] * 3)
        images = images.reshape(-1, 28, 28, 3)
        images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in images])

        num_classes = len(np.unique(labels))

        X = [images]
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
        images = images.reshape(-1, 784)
        images = np.dstack([images] * 3)
        images = images.reshape(-1, 28, 28, 3)
        images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in images])

        num_classes = len(np.unique(labels))

        X = [images]
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
        learning_rate = self._learning_rate
        model = keras.applications.VGG16(
            include_top=True,
            input_shape=(48, 48, 3),
            weights=None, 
            classes=num_classes
        )

        model.compile(
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

# if __name__ == '__main__':
#     knobs = {
#         'batch_size': 8,
#         'epochs': 1,
#         'learning_rate': 1e-5
#     }

#     model = VGG16()
#     model.init(knobs)
#     model.train('tf-keras://fashion_mnist?train_or_test=train')
#     accuracy = model.evaluate('tf-keras://fashion_mnist?train_or_test=test') 
#     print(accuracy)