import numpy as np
from time import time
import os
import base64
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        FixedKnob, CategoricalKnob, dataset_utils, logger
from rafiki.constants import TaskType, ModelDependency

class LeNet5(BaseModel):
    '''
    Implements LeNet5 network to train model for simple image classification
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(10),
            'batch_size': FixedKnob(128)
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._model = self._build_classifier(self.epochs, self.batch_size)

    def train(self, dataset_uri):
        
        ep = self._knobs.get('epochs')
        bs = self._knobs.get('batch_size')
        
        dataset = dataset_utils.load_dataset_of_image_files(dataset_uri, image_size=[28,28])
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

    def evaluate (self, dataset_uri):
        dataset = dataset_utils.load_dataset_of_image_files(dataset_uri, image_size=[28,28])
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        images = self._prepare_X(images)
        images = np.pad(images, ((0,0),(2,2),(2,2),(0,0)), 'constant')

        X_test, y_test = images, to_categorical(classes, num_classes = 10)

        # Compute test accuracy
        (test_loss, test_acc) = self._model.evaluate(X_test, y_test)
        return test_acc
    
    def predict(self, queries):
        X = self._prepare_X(queries)
        X = np.pad(X, ((0,0),(2,2),(2,2),(0,0)), 'constant')
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
        model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(units=120, activation='relu'))
        model.add(layers.Dense(units=84, activation='relu'))
        model.add(layers.Dense(units=10, activation = 'softmax'))
        
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
        return model

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='LeNet5',
        task=TaskType.IMAGE_CLASSIFICATION,
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0',
            ModelDependency.KERAS: '2.2.4'
        },
    train_dataset_uri='data/fashion_mnist_for_image_classification_train.zip',
    test_dataset_uri='data/fashion_mnist_for_image_classification_test.zip',
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



