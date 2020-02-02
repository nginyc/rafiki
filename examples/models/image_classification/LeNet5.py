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

import numpy as np
from time import time
import os
import base64
import pickle
import tensorflow as tf
from keras import models, layers
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

from rafiki.model import BaseModel, FixedKnob, FloatKnob, CategoricalKnob, utils
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

class LeNet5(BaseModel):
    '''
    Implements LeNet5 network to train image classification model on mnist dataset
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(15),
            'batch_size': CategoricalKnob([32, 64, 128]),
            'l_rate': FloatKnob(0.0001, 0.001, 0.01),
            'max_image_size': CategoricalKnob([28, 32])
        }


    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._model = self._build_classifier(self.l_rate)

  
    def train(self, dataset_path, **kwargs):
        ep = self._knobs.get('epochs')
        bs = self._knobs.get('batch_size')
        
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=self.max_image_size, mode='L')
        self._image_size = dataset.image_size
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        images = self._prepare_X(images)
        train = {}
        train['images'] = images
        train['classes'] = classes
        validation = {}
        train['images'], validation['images'], train['classes'], validation['classes'] = train_test_split(train['images'], train['classes'], test_size=0.2, random_state=0)    
        train['images'] =  np.pad(train['images'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
        validation['images'] = np.pad(validation['images'], ((0,0),(2,2),(2,2),(0,0)), 'constant')

        X_train, y_train = train['images'], to_categorical(train['classes'], num_classes = 10)
        X_validation, y_validation = validation['images'], to_categorical(validation['classes'])

        train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=bs)
        validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=bs)
        steps_per_epoch = X_train.shape[0]//bs
        validation_steps = X_validation.shape[0]//bs

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self._model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=ep, 
                    validation_data=validation_generator, validation_steps=validation_steps, 
                    shuffle=True, callbacks=[tensorboard])

        # Compute train accuracy
        (train_loss, train_acc) = self._model.evaluate(X_validation, y_validation)
        utils.logger.log('Train loss: {}'.format(train_loss))
        utils.logger.log('Train accuracy: {}'.format(train_acc))


    def evaluate (self, dataset_path):
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=self.max_image_size, mode='L')        
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        images = self._prepare_X(images)
        images = np.pad(images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        X_test, y_test = images, to_categorical(classes, num_classes = 10)

        # Compute test accuracy
        (test_loss, test_acc) = self._model.evaluate(X_test, y_test)
        return test_acc


    def predict(self, queries):
        queries = utils.dataset.transform_images(queries, image_size=self._image_size, mode='L')
        X = self._prepare_X(queries)
        X = np.pad(X, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        probs = self._model.predict_proba(X)
        
        return probs.tolist()

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}
        # Put model parameters
        model_bytes = pickle.dumps(self._model)
        model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        params['model_base64'] = model_base64

        # Put image size
        params['image_size'] = self._image_size

        return params

    def load_parameters(self, params):
        # Load model parameters
        model_base64 = params['model_base64']
        model_bytes = base64.b64decode(model_base64.encode('utf-8'))
        self._model = pickle.loads(model_bytes)

        # Load image size
        self._image_size = params['image_size']

    def _prepare_X(self, images):
        X = np.asarray(images)
        return X.reshape(-1,28,28,1)
            
    
    def _build_classifier(self, l_rate):
        l_rate = self._knobs.get('l_rate')
        model = models.Sequential()
        model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(units=120, activation='relu'))
        model.add(layers.Dense(units=84, activation='relu'))
        model.add(layers.Dense(units=10, activation = 'softmax'))      
        adam = Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
        
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
        model_class='LeNet5',
        task='IMAGE_CLASSIFICATION',
        dependencies={
            ModelDependency.KERAS: '2.2.4'
            },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries

    )
