import abc
import os
import re
import logging
import urllib
import pandas as pd
import numpy as np
from tensorflow import keras

from .BasePreparator import BasePreparator

class NoSuchKerasDatasetException(Exception):
    pass

class TensorflowKerasDatasetPreparator(BasePreparator):
    def __init__(self, keras_dataset_name):
        '''
        Extracts a dataset from `keras.datasets`
        Args:
            keras_dataset_name - Dataset name in `keras.datasets`
        '''
        if not hasattr(keras.datasets, keras_dataset_name):
            raise NoSuchKerasDatasetException(
                'keras.datasets.{} does not exist.'.format(keras_dataset_name)
            )
        
        self._dataset = getattr(keras.datasets, keras_dataset_name)

    def transform_data(self, queries, labels=None):
        '''
        Args:
            queries - iterable of numpy arrays of shape (None, None) as queries
            labels - iterable of ints as labels
        '''
        X = np.array(queries)
        y = np.array(labels)

        return X, y
    
    def reverse_transform_data(self, X=None, y=None):
        if X is not None:
            X = X.tolist()
            
        if y is not None:
            y = [int(x) for x in y]

        return X, y

    def get_train_data(self):
        (train_images, train_labels), (test_images, test_labels) = self._dataset.load_data()
        X, y = self.transform_data(train_images, train_labels)
        return X, y

