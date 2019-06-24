import os
import uuid
import shutil

from .data_store import DataStore, Dataset

class FileDataStore(DataStore):
    '''
       Stores datasets in the local filesystem.
    '''

    def __init__(self, data_dir=None):
        self._data_dir = data_dir or os.path.join(os.environ['WORKDIR_PATH'], os.environ['DATA_DIR_PATH'])

    def save(self, data_file_path):
        # Copy file to data dir
        file_name = '{}.data'.format(uuid.uuid4())
        dest_file_path = os.path.join(self._data_dir, file_name)
        shutil.copyfile(data_file_path, dest_file_path)
        
        # Get metadata for dataset
        dataset_id = file_name
        size_bytes = self._get_size_bytes(dest_file_path)

        return Dataset(dataset_id, size_bytes)
    
    def load(self, dataset_id):
        file_name = dataset_id
        file_path = os.path.join(self._data_dir, file_name)
        return file_path
