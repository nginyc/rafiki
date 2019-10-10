#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import requests
import json
import pickle
import os
from functools import wraps
from typing import Type, Dict, List, Any

from rafiki.constants import ModelAccessRight, ModelDependencies, Budget, BudgetOption, \
                            InferenceBudget, InferenceBudgetOption, UserType
from rafiki.model import Params, BaseModel

class RafikiConnectionError(ConnectionError): pass

DOCS_URL = 'https://nginyc.github.io/rafiki/docs/latest/docs/src/python/rafiki.client.Client.html'

# Returns a decorator that warns user about the method being deprecated
def _deprecated(msg=None):
    def deco(func):
        nonlocal msg
        msg = msg or f'`{func.__name__}` has been deprecated.'

        @wraps(func)
        def deprecated_func(*args, **kwargs):
            _warn(f'{msg}\n' \
                f'Refer to the updated documentation at {DOCS_URL}')
            return func(*args, **kwargs)
        
        return deprecated_func
    return deco

class Client():

    '''
    Initializes the Client to connect to a running 
    Rafiki Admin instance that the Client connects to.

    :param admin_host: Host of Rafiki Admin
    :param admin_port: Port of Rafiki Admin
    '''
    def __init__(self, admin_host: str = os.environ.get('RAFIKI_ADDR', 'localhost'),
                    admin_port: int = os.environ.get('ADMIN_EXT_PORT', 3000)):
        self._admin_host = admin_host
        self._admin_port = admin_port
        self._token = None
        self._user = None

    def login(self, email: str, password: str) -> Dict[str, Any]:
        '''
        Creates a login session as a Rafiki user. You will have to be logged in to perform any actions.

        App developers can create, list and stop train and inference jobs, as well as list models.
        Model developers can create and list models.

        The login session (the session token) expires in 1 hour.

        :param email: User's email
        :param password: User's password

        :returns: Logged-in user as dictionary
        '''
        data = self._post('/tokens', json={
            'email': email,
            'password': password
        })
        self._token = data['token']

        # Save user's data
        self._user = {
            'id': data['user_id'],
            'user_type': data['user_type']
        }

        return self._user

    def get_current_user(self) -> Dict[str, Any]:
        '''
        Gets currently logged in user's data.

        :returns: Current user as dictionary, or ``None`` if client is not logged in
        '''
        return self._user

    def logout(self):
        '''
        Clears the current login session.
        '''
        self._token = None
        self._user = None

    ####################################
    # User
    ####################################

    def create_user(self, email: str, password: str, user_type: UserType) -> Dict[str, Any]:
        '''
        Creates a Rafiki user. 
        
        Only admins can create users (except for admins).
        Only superadmins can create admins.

        :param email: The new user's email
        :param password: The new user's password
        :param user_type: The new user's type

        :returns: Created user as dictionary
        '''
        data = self._post('/users', json={
            'email': email,
            'password': password,
            'user_type': user_type
        })
        return data

    @_deprecated('`create_users` has been removed')
    def create_users(self, *args, **kwargs):
        pass

    def get_users(self) -> List[Dict[str, Any]]:
        '''
        Lists all Rafiki users.
        
        Only admins can list all users.

        :returns: List of users as list of dictionaries
        '''
        data = self._get('/users')
        return data

    def ban_user(self, email: str) -> Dict[str, Any]:
        '''
        Bans a Rafiki user, disallowing logins.
        
        This action is irrevisible.
        Only admins can ban users (except for admins).
        Only superadmins can ban admins.

        :param email: The user's email

        :returns: Banned user as dictionary
        '''
        data = self._delete('/users', json={
            'email': email
        })
        return data

    ####################################
    # Datasets
    ####################################

    def create_dataset(self, name: str, task: str, dataset_path: str = None, dataset_url: str = None) -> Dict[str, Any]:
        '''
        Creates a dataset on Rafiki, either by uploading the dataset file from your filesystem or specifying a URL where the dataset file can be downloaded.
        The dataset should be in a format specified by the task
        Either `dataset_url` or `dataset_path` should be specified.

        Only admins, model developers and app developers can manage their own datasets.

        :param name: Name for the dataset, does not need to be unique
        :param task: Task associated to the dataset
        :param dataset_path: Path to the dataset file to upload from the local filesystem
        :param dataset_url: Publicly accessible URL where the dataset file can be downloaded
        :returns: Created dataset as dictionary
        '''
        files = {}

        if dataset_path is not None:
            f = open(dataset_path, 'rb')
            dataset = f.read()
            f.close()
            files['dataset'] = dataset
            print('Uploading dataset..')
        else:
            print('Waiting for server finish downloading the dataset from URL...')

        data = self._post(
            '/datasets', 
            files=files,
            form_data={
                'name': name,
                'task': task,
                'dataset_url': dataset_url
            }
        )
        return data

    def get_datasets(self, task: str = None) -> List[Dict[str, Any]]:
        '''
        Lists all datasets owned by the current user, optionally filtering by task.

        :param task: Task name
        :returns: List of datasets as list of dictionaries
        '''
        data = self._get('/datasets', params={
            'task': task
        })
        return data

    ####################################
    # Models
    ####################################

    def create_model(self, name: str, task: str, model_file_path: str, model_class: str, dependencies: ModelDependencies = None, 
                    access_right: ModelAccessRight = ModelAccessRight.PRIVATE, docker_image: str = None) -> Dict[str, Any]:
        '''
        Creates a model on Rafiki.

        Only admins & model developers can manage models.

        :param name: Name of the model, which must be unique across all models added by the current user 
        :param task: Task associated with the model, where the model must adhere to the specification of the task
        :param model_file_path: Path to a single Python file that contains the definition for the model class
        :param model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
        :param dependencies: List of Python dependencies & their versions
        :param access_right: Model access right
        :param docker_image: A custom Docker image that extends ``rafikiai/rafiki_worker``, publicly available on Docker Hub.
        :returns: Created model as dictionary

        Refer to :ref:`model-development` for more details on how to write & test models for Rafiki.

        ``model_file_path`` should point to a *single* file that contains all necessary Python code for the model's implementation. 
        If the Python file imports any external Python modules, you should list it in ``dependencies`` or create a custom
        ``docker_image``. 

        If a model's ``access_right`` is set to ``PUBLIC``, this model will be publicly available to all other users on Rafiki for training
        and inference. By default, a model's access is ``PRIVATE``.

        ``dependencies`` should be a dictionary of ``{ <dependency_name>: <dependency_version> }``, where 
        ``<dependency_name>`` corresponds to the name of the Python Package Index (PyPI) package (e.g. ``tensorflow``)
        and ``<dependency_version>`` corresponds to the version of the PyPI package (e.g. ``1.12.0``). 
        Refer to :ref:`configuring-model-environment` to understand more about this option.
        '''
        f = open(model_file_path, 'rb')
        model_file_bytes = f.read()
        f.close()
        
        data = self._post(
            '/models', 
            files={
                'model_file_bytes': model_file_bytes
            },
            form_data={
                'name': name,
                'task': task,
                'dependencies': json.dumps(dependencies),
                'docker_image': docker_image,
                'model_class':  model_class,
                'access_right': access_right
            }
        )
        return data

    def get_model(self, model_id: str) -> Dict[str, Any]:
        '''
        Retrieves details of a single model.

        Model developers can only view their own models.

        :param model_id: ID of model
        :returns: Model as dictionary
        '''
        _note('`get_model` now requires `model_id` instead of `name`')

        data = self._get('/models/{}'.format(model_id))
        return data

    def download_model_file(self, model_id: str, out_model_file_path: str) -> Dict[str, any]:
        '''
        Downloads the Python model class file for the Rafiki model.

        Model developers can only download their own models.

        :param model_id: ID of model
        :param out_model_file_path: Absolute/relative path to save model class file to
        :returns: Model as dictionary
        '''
        _note('`download_model_file` now requires `model_id` instead of `name`')

        model_file_bytes = self._get('/models/{}/model_file'.format(model_id))

        with open(out_model_file_path, 'wb') as f:
            f.write(model_file_bytes)

        data = self.get_model(model_id)
        dependencies = data.get('dependencies')
        model_class = data.get('model_class')

        print('Model file downloaded to "{}"!'.format(os.path.join(os.getcwd(), out_model_file_path)))
        
        if dependencies:
            print('You\'ll need to install the following model dependencies locally: {}'.format(dependencies))

        print('From the file, import the model class `{}`.'.format(model_class))

        return data

    @_deprecated('`get_models` & `get_models_of_task` have been combined into `get_available_models`')
    def get_models(self, *args, **kwargs):
        pass

    @_deprecated('`get_models` & `get_models_of_task` have been combined into `get_available_models`')
    def get_models_of_task(self, *args, **kwargs):
        pass

    def get_available_models(self, task: str = None) -> List[Dict[str, Any]]:
        '''
        Lists all Rafiki models available to the current user, optionally filtering by task.

        :param task: Task name
        :returns: Available models as list of dictionaries
        '''
        data = self._get('/models/available', params={
            'task': task
        })
        return data

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        '''
        Deletes a single model. Models that have been used in train jobs cannot be deleted.

        Model developers can only delete their own models.

        :param str model_id: ID of model
        :returns: Deleted model as dictionary
        '''
        data = self._delete('/models/{}'.format(model_id))
        return data

    ####################################
    # Train Jobs
    ####################################
    
    def create_train_job(self, app: str, task: str, train_dataset_id: str, val_dataset_id: str, 
                        budget: Budget, models: List[str] = None, train_args: Dict[str, any] = None) -> Dict[str, Any]:
        '''
        Creates and starts a train job on Rafiki. 

        A train job is uniquely identified by user, its associated app, and the app version (returned in output).
        
        Only admins, model developers & app developers can manage train jobs. Model developers & app developers can only manage their own train jobs.

        :param app: Name of the app associated with the train job
        :param task: Task associated with the train job, 
            the train job will train models associated with the task
        :param train_dataset_id: ID of the train dataset, previously created on Rafiki
        :param val_dataset_id: ID of the validation dataset, previously created on Rafiki
        :param budget: Budget for train job
        :param models: List of IDs of model to use for train job. Defaults to all available models
        :param train_args: Additional arguments to pass to models during training, if any. 
            Refer to the task's specification for appropriate arguments  
        :returns: Created train job as dictionary

        If ``models`` is unspecified, all models accessible to the user for the specified task will be used.

        ``budget`` should be a dictionary of ``{ <budget_type>: <budget_amount> }``, where 
        ``<budget_type>`` is one of :class:`rafiki.constants.BudgetOption` and 
        ``<budget_amount>`` specifies the amount for the associated budget option.
        
        The following describes the budget options available:

        =====================       =====================
        **Budget Option**             **Description**
        ---------------------       ---------------------
        ``TIME_HOURS``              Max no. of hours to train (soft target). Defaults to 0.1.
        ``GPU_COUNT``               No. of GPUs to allocate for training, across all models. Defaults to 0.
        ``MODEL_TRIAL_COUNT``       Max no. of trials to conduct for each model (soft target). -1 for unlimited. Defaults to -1.
        =====================       =====================
        '''
        _note('`create_train_job` now requires `models` as a list of model IDs instead of a list of model names')
        
        if 'ENABLE_GPU' in budget:
            _warn('The `ENABLE_GPU` option has been changed to `GPU_COUNT`')
            
        # Default to all available models
        if models is None: 
            avail_models = self.get_available_models(task)
            models = [x['id'] for x in avail_models]

        # Have defaults for budget
        budget = {
            BudgetOption.TIME_HOURS: 0.1,
            BudgetOption.GPU_COUNT: 0,
            **budget
        }

        data = self._post('/train_jobs', json={
            'app': app,
            'task': task,
            'train_dataset_id': train_dataset_id,
            'val_dataset_id': val_dataset_id,
            'budget': budget,
            'model_ids': models,
            'train_args': train_args
        })
        return data

    def get_train_jobs_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        '''
        Lists all of user's train jobs on Rafiki.

        :param user_id: ID of the user
        :returns: Train jobs as list of dictionaries
        '''
        data = self._get('/train_jobs', params={ 
            'user_id': user_id
        })
        return data
    
    def get_train_jobs_of_app(self, app: str) -> List[Dict[str, Any]]:
        '''
        Lists all of current user's train jobs associated to the app name on Rafiki.

        :param app: Name of the app
        :returns: Train jobs as list of dictionaries
        '''
        data = self._get('/train_jobs/{}'.format(app))
        return data

    def get_train_job(self, app: str, app_version: int = -1) -> Dict[str, Any]:
        '''
        Retrieves details of the current user's train job identified by an app and an app version, 
        including workers' details.

        :param app: Name of the app
        :param app_version: Version of the app (-1 for latest version)
        :returns: Train job as dictionary
        '''
        data = self._get('/train_jobs/{}/{}'.format(app, app_version))
        return data

    def stop_train_job(self, app: str, app_version: int = -1) -> Dict[str, Any]:
        '''
        Prematurely stops the current user's train job identified by an app and an app version.
        Otherwise, the train job should stop by itself when its budget is reached.

        :param app: Name of the app
        :param app_version: Version of the app (-1 for latest version)
        :returns: Stopped train job as dictionary
        '''
        data = self._post('/train_jobs/{}/{}/stop'.format(app, app_version))
        return data

    ####################################
    # Trials
    ####################################

    def get_trial(self, trial_id: str) -> Dict[str, Any]:
        '''
        Gets a specific trial.

        :param trial_id: ID of trial
        :returns: Trial as dictionary
        '''
        data = self._get('/trials/{}'.format(trial_id))
        return data

    def get_best_trials_of_train_job(self, app: str, app_version: int = -1, max_count: int = 2) -> List[Dict[str, Any]]:
        '''
        Lists the best scoring trials of the current user's train job identified by an app and an app version,
        ordered by descending score.

        :param app: Name of the app
        :param app_version: Version of the app (-1 for latest version)
        :param max_count: Maximum number of trials to return
        :returns: Trials as list of dictionaries
        '''
        data = self._get('/train_jobs/{}/{}/trials'.format(app, app_version), params={
            'type': 'best',
            'max_count': max_count
        })
        return data

    def get_trials_of_train_job(self, app: str, app_version: int = -1) -> List[Dict[str, Any]]:
        '''
        Lists all trials of the current user's train job identified by an app and an app version,
        ordered by when the trial started.

        :param app: Name of the app
        :param app_version: Version of the app (-1 for latest version)
        :returns: Trials as list of dictionaries
        '''
        data = self._get('/train_jobs/{}/{}/trials'.format(app, app_version))
        return data

    def get_trial_logs(self, trial_id: str) -> Dict[str, Any]:
        '''
        Gets the logs for a specific trial.

        :param trial_id: ID of trial
        :returns: Logs of trial as dictionary
        '''
        data = self._get('/trials/{}/logs'.format(trial_id))
        return data

    def get_trial_parameters(self, trial_id: str) -> Params:
        '''
        Gets parameters of the model associated with the trial. The trial's model parameters must have been saved.

        :param trial_id: ID of trial
        :returns: Parameters of the *trained* model associated with the trial
        '''
        data = self._get('/trials/{}/parameters'.format(trial_id))
        parameters = pickle.loads(data)
        return parameters

    def load_trial_model(self, trial_id: str, ModelClass: Type[BaseModel]) -> BaseModel:
        '''
        Loads an instance of a trial's model with the trial's knobs & parameters.

        Before this, you must have the trial's model class file already in your local filesystem,
        the dependencies of the model must have been installed separately, and the model class must have been 
        imported and passed into this method.

        Wraps :meth:`get_trial_parameters` and :meth:`get_trial`.

        :param trial_id: ID of trial
        :param ModelClass: model class that conincides with the trial's model class
        :returns: A *trained* model instance of ``ModelClass``, loaded with the trial's knobs and parameters
        '''
        data = self.get_trial(trial_id)
        assert 'proposal' in data
        knobs = data['proposal']['knobs']
        parameters = self.get_trial_parameters(trial_id)
        model_inst = ModelClass(**knobs)
        model_inst.load_parameters(parameters)
        return model_inst

    ####################################
    # Inference Jobs
    ####################################

    def create_inference_job(self, app: str, app_version: int = -1, budget: InferenceBudget = None) -> Dict[str, Any]:
        '''
        Creates and starts a inference job on Rafiki with the best-scoring trials of the associated train job. 
        The train job must have the status of ``STOPPED``.The inference job would be tagged with the train job's app and app version. 
        Throws an error if an inference job of the same train job is already running.

        In this method's response, `predictor_host` is this inference job's predictor's host. 

        Only admins, model developers & app developers can manage inference jobs. Model developers & app developers can only manage their own inference jobs.

        :param app: Name of the app identifying the train job to use
        :param app_version: Version of the app identifying the train job to use
        :param budget: Budget for inference job
        :returns: Created inference job as dictionary

        ``budget`` should be a dictionary of ``{ <budget_type>: <budget_amount> }``, where 
        ``<budget_type>`` is one of :class:`rafiki.constants.InferenceBudgetOption` and 
        ``<budget_amount>`` specifies the amount for the associated budget option.
        
        The following describes the budget options available:

        =====================       =====================
        **Budget Option**             **Description**
        ---------------------       ---------------------
        ``GPU_COUNT``               No. of GPUs to allocate for inference, across all trials. Defaults to 0.
        =====================       =====================
        '''

        # Have defaults for budget
        budget = {
            InferenceBudgetOption.GPU_COUNT: 0,
            **(budget or {})
        }

        data = self._post('/inference_jobs', json={
            'app': app,
            'app_version': app_version,
            'budget': budget
        })
        return data

    def get_inference_jobs_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        '''
        Lists all of user's inference jobs on Rafiki.

        :param user_id: ID of the user
        :returns: Inference jobs as list of dictionaries
        '''
        data = self._get('/inference_jobs', params={ 
            'user_id': user_id
        })
        return data

    def get_inference_jobs_of_app(self, app: str) -> List[Dict[str, Any]]:
        '''
        Lists all inference jobs associated to an app on Rafiki.

        :param app: Name of the app
        :returns: Inference jobs as list of dictionaries
        '''
        data = self._get('/inference_jobs/{}'.format(app))
        return data

    def get_running_inference_job(self, app: str, app_version: int = -1) -> Dict[str, Any]:
        '''
        Retrieves details of the *running* inference job identified by an app and an app version,
        including workers' details.

        :param app: Name of the app 
        :param app_version: Version of the app (-1 for latest version)
        :returns: Inference job as dictionary
        '''
        data = self._get('/inference_jobs/{}/{}'.format(app, app_version))
        return data

    def stop_inference_job(self, app: str, app_version: int = -1) -> Dict[str, Any]:
        '''
        Stops the inference job identified by an app and an app version.

        :param app: Name of the app
        :param app_version: Version of the app (-1 for latest version)
        :returns: Stopped inference job as dictionary
        '''
        data = self._post('/inference_jobs/{}/{}/stop'.format(app, app_version))
        return data

    # TODO: Add predict method?

    ####################################
    # Administrative
    ####################################

    def stop_all_jobs(self):
        '''
        Stops all train and inference jobs on Rafiki. 

        Only the superadmin can call this.
        '''
        data = self._post('/actions/stop_all_jobs')
        return data

    ####################################
    # Rafiki Internal
    ####################################

    def send_event(self, name, **params):
        data = self._post('/event/{}'.format(name), json=params)
        return data

    ####################################
    # Private
    ####################################

    def _get(self, path, params=None):
        url = self._make_url(path)
        headers = self._get_headers()
        res = requests.get(
            url,
            headers=headers,
            params=params or {}
        )
        return self._parse_response(res)

    def _post(self, path, params=None, files=None, form_data=None, json=None):
        url = self._make_url(path)
        headers = self._get_headers()
        res = requests.post(
            url, 
            headers=headers,
            params=params or {}, 
            data=form_data,
            json=json,
            files=files or {}
        )
        return self._parse_response(res)

    def _delete(self, path, params=None, files=None, form_data=None, json=None):
        url = self._make_url(path)
        headers = self._get_headers()
        res = requests.delete(
            url, 
            headers=headers,
            params=params or {}, 
            data=form_data or {},
            json=json,
            files=files
        )
        return self._parse_response(res)

    def _make_url(self, path):
        url = 'http://{}:{}{}'.format(self._admin_host, self._admin_port, path)
        return url

    def _parse_response(self, res):
        if res.status_code != 200:
            raise RafikiConnectionError(res.text)

        content_type = res.headers.get('content-type')
        if content_type == 'application/json':
            return res.json()
        elif content_type == 'application/octet-stream':
            return res.content
        else:
            raise RafikiConnectionError('Invalid response content type: {}'.format(content_type))

    def _get_headers(self):
        if self._token is not None:
            return {
                'Authorization': 'Bearer ' + self._token
            }
        else:
            return {}

def _warn(msg):
    print(f'\033[93mWARNING: {msg}\033[0m')

def _note(msg):
    print(f'\033[94m{msg}\033[0m')