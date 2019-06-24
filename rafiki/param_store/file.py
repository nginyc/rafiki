import os
import uuid

from .param_store import ParamStore, Params

class FileParamStore(ParamStore):
    '''
       Stores parameters in the local filesystem.
    '''
    def __init__(self, params_dir=None):
        self._params_dir = params_dir or os.path.join(os.environ['WORKDIR_PATH'], os.environ['PARAMS_DIR_PATH'])

    def save(self, params: Params):
        # Serialize params and save bytes to params dir
        file_name = '{}.model'.format(uuid.uuid4())
        dest_file_path = os.path.join(self._params_dir, file_name)
        params_bytes = self._serialize_params(params)
        with open(dest_file_path, 'wb') as f:
            f.write(params_bytes)
        
        # ID for params is its file name
        params_id = file_name

        return params_id
    
    def load(self, params_id):
        # Load bytes to params dir and deserialize params
        file_name = params_id
        file_path = os.path.join(self._params_dir, file_name)
        with open(file_path, 'rb') as f:
            params_bytes = f.read()
        params = self._deserialize_params(params_bytes)

        return params
