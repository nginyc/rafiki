import os
import abc
from urllib.parse import urlparse, parse_qs
from tensorflow import keras

def load_dataset(dataset_uri):
    parse_result = urlparse(dataset_uri)

    if parse_result.scheme == 'tf-keras':
        dataset_name = parse_result.netloc
        query = parse_qs(parse_result.query)
        train_or_test = query.get('train_or_test', ['train'])[0]
        return load_tf_keras_dataset(dataset_name, train_or_test)
    else:
        raise Exception('Dataset URI scheme not supported: {}'.format(parse_result.scheme))


def load_tf_keras_dataset(dataset_name, train_or_test):
    keras_dataset = getattr(keras.datasets, dataset_name)

    (train_images, train_labels), (test_images, test_labels) = \
        keras_dataset.load_data()

    if train_or_test == 'train':
        return (train_images, train_labels)
    elif train_or_test == 'test':
        return (test_images, test_labels)
    else:
        raise Exception('Invalid `train_or_test` value: {}'.format(train_or_test))
        