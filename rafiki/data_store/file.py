import os
import uuid

from .data_store import DataStore, Dataset

class FileDataStore(DataStore):
    '''
       Stores datasets in the filesystem.
    '''

    def __init__(self, data_dir):
        self._data_dir = data_dir

    def save(self, data_bytes):
        # Save file to disk
        file_name = '{}.data'.format(uuid.uuid4())
        file_path = os.path.join(self._data_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(data_bytes)
        
        # Get metadata for dataset
        dataset_id = file_name
        size_bytes = self._get_size_bytes(data_bytes)

        return Dataset(dataset_id, size_bytes)
    
    def load(self, dataset_id):
        file_name = dataset_id
        file_path = os.path.join(self._data_dir, file_name)
        
        with open(file_path, 'rb') as f:
            data_bytes = f.read
        
        return data_bytes
