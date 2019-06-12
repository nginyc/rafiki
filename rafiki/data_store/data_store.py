import abc
import sys
from typing import List, Dict
import os

class Dataset():
    def __init__(self, id: str, size_bytes: int):
        self.id = id
        self.size_bytes = size_bytes

class DataStore(abc.ABC):
    '''
        Persistent store for datasets.
    '''

    @abc.abstractclassmethod
    def save(self, data_file_path: str) -> Dataset:
        '''
            Persists a dataset in the local filesystem at file path, returning a ``Dataset`` abstraction containing a unique ID for the dataset.
        '''
        raise NotImplementedError()

    @abc.abstractclassmethod
    def load(self, dataset_id: str) -> str:
        '''
            Loads a persisted dataset to the local filesystem, identified by ID, returning the file path to the dataset.
        '''
        raise NotImplementedError()

    @staticmethod
    def _get_size_bytes(data_file_path):
        st = os.stat(data_file_path)
        return st.st_size

    