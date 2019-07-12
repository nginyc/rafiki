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
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import models
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        CategoricalKnob, FixedKnob, dataset_utils, logger
from rafiki.constants import TaskType, ModelDependency

class CNN(BaseModel):
    '''
    Implements CNN using keras for simple image classification
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(50),
            'batch_size': CategoricalKnob([16,32,64,86])
        }


    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._model = self._build_classifier(self.epochs, self.batch_size)


    def train(self, dataset_path):
        ep = self._knobs.get('epochs')
        bs = self._knobs.get('batch_size')       
        dataset = dataset_utils.load_dataset_of_image_files(dataset_path, image_size=[28,28])
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])          
        train = {}
        train['images'] = self._prepare_X(images)
        train['classes'] = classes
        validation = {}
        train['images'], validation['images'], train['classes'], validation['classes'] = train_test_split(train['images'], train['classes'], test_size=0.15, random_state=0)
        X_train, y_train = train['images'], to_categorical(train['classes'], num_classes = 10)
        random_seed = 2
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state=random_seed)     
        datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             fill_mode='nearest')  
        datagen.fit(X_train)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
        history = self._model.fit_generator(datagen.flow(X_train, y_train, batch_size=bs),
                              epochs=ep, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // bs,
                              callbacks=[learning_rate_reduction]) 
        
        # Compute train accuracy
        (train_loss, train_acc) = self._model.evaluate(X_val, y_val)
        logger.log('Train loss: {}'.format(train_loss))
        logger.log('Train accuracy: {}'.format(train_acc))


    def evaluate (self, dataset_path):
        dataset = dataset_utils.load_dataset_of_image_files(dataset_path, image_size=[28,28])
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        images = self._prepare_X(images)
        X_test, y_test = images, to_categorical(classes, num_classes = 10)
        
        # Compute test accuracy
        (test_loss, test_acc) = self._model.evaluate(X_test, y_test)
        return test_acc


    def predict(self, queries):
        X = self._prepare_X(queries)
        probs = self._model.predict_proba(X)
        return probs.tolist()


    def destroy(self):
        pass


    def dump_parameters(self):
        params = {}
        # Save model parameters
        model_bytes = pickle.dumps(self._model)
        model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        params['model_base64'] = model_base64
        return params


    def load_parameters(self, params):
        # Load model parameters
        model_base64 = params.get('model_base64', None)
        if model_base64 is None:
                raise InvalidModelParamsException()
                model_bytes = base64.b64decode(params['model_base64'].encode('utf-8'))
                self._model = pickle.loads(model_bytes)


    def _prepare_X(self, images):
        X = np.asarray(images)
        return X.reshape(-1,28,28,1)


    def _build_classifier(self, epochs, batch_size):
        model = models.Sequential()
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation = "relu"))
        model.add(BatchNormalization())
        model.add(Dense(10, activation = "softmax"))
        my_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model


if __name__ == '__main__':
    test_model_class(
    model_file_path=__file__,
    model_class='CNN',
    task=TaskType.IMAGE_CLASSIFICATION,
    dependencies={
        ModelDependency.TENSORFLOW: '1.12.0',
        ModelDependency.KERAS: '2.2.4'
        },
    train_dataset_path='data/fashion_mnist_for_image_classification_train.zip',
    val_dataset_path='data/fashion_mnist_for_image_classification_val.zip',
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

