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

import base64
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import objectives
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, Dropout, Activation, Flatten, Concatenate, Convolution2D, MaxPooling2D
from keras.layers.core import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

from rafiki.model import BaseModel, FixedKnob, FloatKnob, CategoricalKnob, utils
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

class SqueezeNet(BaseModel):
    '''
    Implements SqueezeNet architecture using keras for simple image classification on mnist dataset
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(15),
            'learning_rate': FloatKnob(0.001, 0.07),
            'decay_rate': FloatKnob(5e-5, 1e-4, is_exp=True),
            'momentum': FloatKnob(0.1, 0.3, 0.6),
            'batch_size': CategoricalKnob([32, 64, 128]),
            'max_image_size': FixedKnob(28)

        }    

    
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._model = self._build_classifier(self.learning_rate, self.decay_rate, self.momentum)

        
    def train(self, dataset_path, **kwargs):
        epochs = self._knobs.get('epochs')
        batch_size = self._knobs.get('batch_size')

        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=self.max_image_size, mode='L')
        self._image_size = dataset.image_size
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])   
        X = self._prepare_X(images)
        y = to_categorical(classes, num_classes = 10)
        
        self._model.fit(X, y, nb_epoch=epochs, batch_size=batch_size, verbose=1)

        # Compute train accuracy
        (train_loss, train_acc) = self._model.evaluate(X, y)
        utils.logger.log('Train loss: {}'.format(train_loss))
        utils.logger.log('Train accuracy: {}'.format(train_acc))

                  
    def evaluate (self, dataset_path):
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=self.max_image_size, mode='L')
        self._image_size = dataset.image_size
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])        
        images = self._prepare_X(images)
        X_test, y_test = images, to_categorical(classes, num_classes = 10)
        # Compute test accuracy
        (test_loss, test_acc) = self._model.evaluate(X_test, y_test)
        return test_acc


    def predict(self, queries):
        queries = utils.dataset.transform_images(queries, image_size=self._image_size, mode='L')
        X = self._prepare_X(queries)
        probs = self._model.predict(X)
        return probs.tolist()
                        

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
        images = np.asarray(images)
        X = images.reshape(-1,28,28,1)
        return X
    

    def _build_classifier(self, learning_rate, decay_rate, momentum):
        learning_rate = self._knobs.get('learning_rate')
        decay_rate = self._knobs.get('decay_rate')
        momentum = self._knobs.get('momentum')
        
        sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        
        input_shape=(28,28,1)

        input_img = Input(batch_shape=(None, 28,28,1))
        squeeze=Lambda(lambda x: x ** 2,input_shape=(784,),output_shape=(1,784))(input_img)
        squeeze=Reshape((28,28,1))(squeeze)
        squeeze=Conv2D(64, 3,3,
                                border_mode='valid',
                                input_shape=input_shape)(squeeze)
        squeeze=BatchNormalization()(squeeze)
        squeeze=ELU(alpha=1.0)(squeeze)
        squeeze=MaxPooling2D(pool_size=(2,2))(squeeze)
        squeeze=Conv2D(32, 1, 1,
                                    init='glorot_uniform')(squeeze)
        squeeze=BatchNormalization()(squeeze)
        squeeze=ELU(alpha=1.0)(squeeze)

        squeeze_left=squeeze
        squeeze_left=Conv2D(64, 3,3,
                                border_mode='valid',
                                input_shape=input_shape)(squeeze_left)
        squeeze_left=ELU(alpha=1.0)(squeeze_left)

        squeeze_right=squeeze
        squeeze_right=Conv2D(64, 3,3,
                                border_mode='valid',
                                input_shape=input_shape)(squeeze_right)
        squeeze_right=ELU(alpha=1.0)(squeeze_right)

        squeeze0=Concatenate()([squeeze_left,squeeze_right])
        squeeze0=Dropout(0.2)(squeeze0)
        squeeze0=GlobalAveragePooling2D()(squeeze0)
        squeeze0=Dense(10)(squeeze0)
        squeeze0=Activation('sigmoid')(squeeze0)

        model = Model(inputs = input_img, outputs = squeeze0)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
        model.summary()
        
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
        model_class='SqueezeNet',
        task='IMAGE_CLASSIFICATION',
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0',
            ModelDependency.KERAS: '2.2.4'
            },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries

    )
