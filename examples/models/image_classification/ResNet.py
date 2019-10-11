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
import os
import base64
import pickle
from time import time
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical

from rafiki.model import BaseModel, FixedKnob, FloatKnob, CategoricalKnob, utils
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class


class ResNet(BaseModel):
    '''
    Implements ResNet network to train model for simple image classification
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(15),
            'batch_size': CategoricalKnob([32, 64, 128]),
            'learning_rate': FloatKnob(0.0001, 0.001, is_exp=True),
            'momentum': FloatKnob(0.3, 0.6, 0.9),
            'max_image_size': FixedKnob(224),
            }
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._model = self._build_classifier(self.learning_rate, self.momentum)

  
    def train(self, dataset_path, **kwargs):
        ep = self._knobs.get('epochs')
        bs = self._knobs.get('batch_size')
        max_image_size = self._knobs.get('max_image_size')
        
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, min_image_size=32, 
                                                            max_image_size=max_image_size, mode='RGB')
        self._image_size = dataset.image_size
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        
        images = self._prepare_X(images)
        classes = keras.utils.to_categorical(classes, num_classes=10)
        
        self._model.fit(
                    images, 
                    classes, 
                    epochs=ep, 
                    batch_size=bs
                )

        # Compute train accuracy
        (loss, accuracy) = self._model.evaluate(images, classes)

        utils.logger.log('Train loss: {}'.format(loss))
        utils.logger.log('Train accuracy: {}'.format(accuracy))

        
    def evaluate (self, dataset_path):
        max_image_size = self._knobs.get('max_image_size')
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, min_image_size=32, 
                                                            max_image_size=max_image_size, mode='RGB')
        self._image_size = dataset.image_size
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        
        images = self._prepare_X(images)
        classes = keras.utils.to_categorical(classes, num_classes=10)      
        
        # Compute test accuracy
        (test_loss, test_acc) = self._model.evaluate(images, classes, verbose=1)
        return test_acc


    def predict(self, queries):
        image_size = self._image_size
        images = utils.dataset.transform_images(queries, image_size=image_size, mode='RGB')
        X = self._prepare_X(queries)
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
        images = np.asarray(images)
        images = images/255
        mean = np.mean(images, axis=0)
        images -= mean        
        return images

    
    def _build_classifier(self, learning_rate, momentum):
        lr=self._knobs.get('learning_rate')
        momentum=self._knobs.get('momentum')
        
        base_model = ResNet50(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)     
        predictions = Dense(10, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

        model.compile(optimizer=SGD(lr=lr, momentum=momentum), metrics=['accuracy'], loss='categorical_crossentropy')
                       
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/cifar10_train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/cifar10_val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/cifar10_test.zip', help='Path to test dataset')
    parser.add_argument('--query_path', type=str, default='examples/data/image_classification/cifar10_test_1.png', 
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    test_model_class(
        model_file_path=__file__,
        model_class='ResNet',
        task='IMAGE_CLASSIFICATION',
        dependencies={
                    ModelDependency.KERAS: '2.2.4'        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )