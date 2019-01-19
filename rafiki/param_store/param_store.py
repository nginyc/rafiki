import os

class ParamsExistsError(Exception): pass
class InvalidParamsError(Exception): pass

class ParamStore(object):
    '''
    Store API that reads & writes parameters.
    '''
    def __init__(self, **kwargs):
        self._params_dir_path = kwargs.get('params_dir_path') or \
                                os.path.join(os.environ['WORKDIR_PATH'], os.environ['PARAMS_DIR_PATH'])
    
    '''
    Retrieves parameters from underlying storage.
    Throws `InvalidParamsError` if parameters of ID doesn't exist.

    :param str param_id: ID of parameters
    :returns: parameters
    :rtype: bytes
    '''
    def get_params(self, param_id):
        param_file_path = os.path.join(self._params_dir_path, param_id)

        if not os.path.isfile(param_file_path):
            raise InvalidParamsError('Params of ID "{}" does not exist'.format(param_id))

        parameters = None
        with open(param_file_path, 'rb') as f:
            parameters = f.read()
        
        return parameters

    '''
    Stores parameters into underlying storage.
    Throws `ParamsExistsError` if parameters of ID already exists.

    :param str param_id: ID of parameters (must be unique)
    :param bytes parameters: Parameters to store
    :returns: ID of parameters
    :rtype: str
    '''
    def put_params(self, param_id, parameters):
        param_file_path = os.path.join(self._params_dir_path, param_id)
        
        if os.path.isfile(param_file_path):
            raise ParamsExistsError('Params of ID "{}" already exists'.format(param_id))

        with open(param_file_path, 'wb') as f:
            f.write(parameters)
        
        return param_id
        

