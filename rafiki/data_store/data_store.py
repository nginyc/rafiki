import abc
import sys
from typing import List, Dict

class Dataset():
    def __init__(self, id: str, size_bytes: int):
        self.id = id
        self.size_bytes = size_bytes

class DataStore(abc.ABC):
    '''
        Persistent store for datasets.
    '''

    @abc.abstractclassmethod
    def save(self, data_bytes: bytes) -> Dataset:
        '''
            Saves a dataset as bytes and returns a ``Dataset`` abstraction containing a unique ID for the dataset.
        '''
        raise NotImplementedError()

    @abc.abstractclassmethod
    def load(self, dataset_id: str) -> bytes:
        '''
            Loads the bytes of a dataset identified by ID.
        '''
        raise NotImplementedError()

    @staticmethod
    def _get_size_bytes(data_bytes):
        return sys.getsizeof(data_bytes)

    