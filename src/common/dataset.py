import os
import abc

class DatasetType():
    TF_KERAS = 'TF_KERAS'


class DatasetConfig(abc.ABC):
    dataset_type = None
    params = {}


class TfKerasDatasetConfig(DatasetConfig):
    dataset_type = DatasetType.TF_KERAS

    def __init__(self, dataset_name, train_or_test = 'train'):
        self.params = {
            'dataset_name': dataset_name,
            'train_or_test': train_or_test # 'train' | 'test'
        }


