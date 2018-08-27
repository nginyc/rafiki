import os
import abc
from tensorflow import keras

class DatasetType():
    TF_KERAS = 'TF_KERAS'

def build_tf_keras_dataset_config(dataset_name, train_or_test = 'train'):
    return {
        'dataset_type': DatasetType.TF_KERAS,
        'dataset_name': dataset_name,
        'train_or_test': train_or_test
    }

def load_tf_keras_dataset(dataset_config):
    train_or_test =  dataset_config['train_or_test']
    dataset_name = dataset_config['dataset_name']
    keras_dataset = getattr(keras.datasets, dataset_name)

    (train_images, train_labels), (test_images, test_labels) = \
        keras_dataset.load_data()

    if train_or_test == 'train':
        return (train_images, train_labels)
    elif train_or_test == 'test':
        return (test_images, test_labels)
    else:
        raise Exception('train_or_test should be \'train\' or \'test\'')