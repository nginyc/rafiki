import requests
import json
import pickle
import os
from functools import wraps

from rafiki.constants import ModelAccessRight
from rafiki.constants import Budget, BudgetOption

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

class Client(object):

    '''
    Initializes the Client to connect to a running 
    Rafiki Admin instance that the Client connects to.

    :param str admin_host: Host of Rafiki Admin
    :param int admin_port: Port of Rafiki Admin
    '''
    def __init__(self, admin_host=os.environ.get('RAFIKI_ADDR', 'localhost'),
                    admin_port=os.environ.get('ADMIN_EXT_PORT', 3000)):
        self._admin_host = admin_host
        self._admin_port = admin_port
        self._token = None
        self._user = None

    def login(self, email, password):
        '''
        Creates a login session as a Rafiki user. You will have to be logged in to perform any actions.

        App developers can create, list and stop train and inference jobs, as well as list models.
        Model developers can create and list models.

        The login session (the session token) expires in 1 hour.

        :param str email: User's email
        :param str password: User's password

        :returns: Logged-in user as dictionary
        :rtype: dict[str, any]
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

    def get_current_user(self):
        '''
        Gets currently logged in user's data.

        :returns: Current user as dictionary, or `None` if client is not logged in
        :rtype: dict[str, any]
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

    def create_user(self, email, password, user_type):
        '''
        Creates a Rafiki user. 
        
        Only admins can create users (except for admins).
        Only superadmins can create admins.

        :param str email: The new user's email
        :param str password: The new user's password
        :param user_type: The new user's type
        :type user_type: :class:`rafiki.constants.UserType` 

        :returns: Created user as dictionary
        :rtype: dict[str, any]
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

    def get_users(self):
        '''
        Lists all Rafiki users.
        
        Only admins can list all users.

        :returns: List of users
        :rtype: dict[str, any][]
        '''
        data = self._get('/users')
        return data

    def ban_user(self, email):
        '''
        Bans a Rafiki user, disallowing logins.
        
        This action is irrevisible.
        Only admins can ban users (except for admins).
        Only superadmins can ban admins.

        :param str email: The user's email

        :returns: Banned user as dictionary
        :rtype: dict[str, any]
        '''
        data = self._delete('/users', json={
            'email': email
        })
        return data

    ####################################
    # Datasets
    ####################################

    def create_dataset(self, name, task, dataset_path=None, dataset_url=None):
        '''
        Creates a dataset on Rafiki, either by uploading the dataset file from your filesystem or specifying a URL where the dataset file can be downloaded.
        The dataset should be in a format specified by the task
        Either `dataset_url` or `dataset_path` should be specified.

        Only admins, model developers and app developers can manage their own datasets.

        :param str name: Name for the dataset, does not need to be unique
        :param str task: Task associated to the dataset
        :param str dataset_path: Path to the dataset file to upload from the local filesystem
        :param str dataset_url: Publicly accessible URL where the dataset file can be downloaded
        :returns: Created dataset as dictionary
        :rtype: dict[str, any]
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

    def get_datasets(self, task=None):
        '''
        Lists all datasets owned by the current user, optionally filtering by task.

        :param str trsk: Task name
        :returns: List of datasets
        :rtype: dict[str, any][]
        '''
        data = self._get('/datasets', params={
            'task': task
        })
        return data

    ####################################
    # Models
    ####################################

    def create_model(self, name, task, model_file_path, model_class, dependencies={}, 
                    access_right=ModelAccessRight.PRIVATE, docker_image=None):
        '''
        Creates a model on Rafiki.

        Only admins & model developers can manage models.

        :param str name: Name of the model, which must be unique across all models added by the current user 
        :param str task: Task associated with the model, where the model must adhere to the specification of the task
        :param str model_file_path: Path to a single Python file that contains the definition for the model class
        :param str model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
        :param dependencies: List of dependencies & their versions
        :type dependencies: dict[str, str]
        :param access_right: Model access right
        :type access_right: :class:`rafiki.constants.ModelAccessRight`
        :param str docker_image: A custom Docker image name that extends ``rafikiai/rafiki_worker``
        :returns: Created model as dictionary
        :rtype: dict[str, any]

        ``model_file_path`` should point to a file that contains all necessary Python code for the model's implementation. 
        If the Python file imports any external Python modules, you should list it in ``dependencies`` or create a custom
        ``docker_image``. 

        If a model's ``access_right`` is set to ``PUBLIC``, this model will be publicly available to all other users on Rafiki for training
        and inference. By default, a model's access is ``PRIVATE``.

        ``dependencies`` should be a dictionary of ``{ <dependency_name>: <dependency_version> }``, where 
        ``<dependency_name>`` corresponds to the name of the Python Package Index (PyPI) package (e.g. ``tensorflow``)
        and ``<dependency_version>`` corresponds to the version of the PyPI package (e.g. ``1.12.0``). These dependencies 
        will be lazily installed on top of the worker's Docker image before the submitted model's code is executed.
        If the model is to be run on GPU, Rafiki would map dependencies to their GPU-supported versions, if supported. 
        For example, ``{ 'tensorflow': '1.12.0' }`` will be installed as ``{ 'tensorflow-gpu': '1.12.0' }``.
        Rafiki could also parse specific dependency names to install certain non-PyPI packages. 
        For example, ``{ 'singa': '1.1.1' }`` will be installed as ``singa-cpu=1.1.1`` or ``singa-gpu=1.1.1`` using ``conda``.

        Refer to the list of officially supported dependencies below. For dependencies that are not listed,
        they will be installed as PyPI packages of the specified name and version.

        =====================       =====================
        **Dependency**              **Installation Command**
        ---------------------       ---------------------        
        ``tensorflow``              ``pip install tensorflow==${ver}`` or ``pip install tensorflow-gpu==${ver}``
        ``singa``                   ``conda install -c nusdbsystem singa-cpu=${ver}`` or ``conda install -c nusdbsystem singa-gpu=${ver}``
        ``Keras``                   ``pip install Keras==${ver}``
        ``scikit-learn``            ``pip install scikit-learn==${ver}``
        ``torch``                   ``pip install torch==${ver}``
        =====================       =====================

        Refer to :ref:`creating-models` to understand more about how to write & test models for Rafiki.

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

    def get_model(self, model_id):
        '''
        Retrieves details of a single model.

        Model developers can only view their own models.

        :param str model_id: ID of model
        :returns: Details of model as dictionary
        :rtype: dict[str, any]
        '''
        _note('`get_model` now requires `model_id` instead of `name`')

        data = self._get('/models/{}'.format(model_id))
        return data

    def download_model_file(self, model_id, out_model_file_path):
        '''
        Downloads the Python model class file for the Rafiki model.

        Model developers can only download their own models.

        :param str model_id: ID of model
        :param str out_model_file_path: Absolute/relative path to save model class file to
        :returns: Details of model as dictionary
        :rtype: dict[str, any]
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

    def get_available_models(self, task=None):
        '''
        Lists all Rafiki models available to the current user, optionally filtering by task.

        :param str task: Task name
        :returns: Available models as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/models/available', params={
            'task': task
        })
        return data

    def delete_model(self, model_id):
        '''
        Deletes a single model. Models that have been used in train jobs cannot be deleted.

        Model developers can only delete their own models.

        :param str model_id: ID of model
        :returns: Deleted model
        :rtype: dict[str, any]
        '''
        data = self._delete('/models/{}'.format(model_id))
        return data

    ####################################
    # Train Jobs
    ####################################
    
    def create_train_job(self, app, task, train_dataset_id, val_dataset_id, budget: Budget, models=None):
        '''
        Creates and starts a train job on Rafiki. 

        A train job is uniquely identified by user, its associated app, and the app version (returned in output).
        
        Only admins, model developers & app developers can manage train jobs. Model developers & app developers can only manage their own train jobs.

        :param str app: Name of the app associated with the train job
        :param str task: Task associated with the train job, 
            the train job will train models associated with the task
        :param str train_dataset_id: ID of the train dataset, previously created on Rafiki
        :param str val_dataset_id: ID of the validation dataset, previously created on Rafiki
        :param str budget: Budget for each model
        :param str[] models: List of IDs of model to use for train job. Defaults to all available models
        :returns: Created train job as dictionary
        :rtype: dict[str, any]

        If ``models`` is unspecified, all models accessible to the user for the specified task will be used.

        ``budget`` should be a dictionary of ``{ <budget_type>: <budget_amount> }``, where 
        ``<budget_type>`` is one of :class:`rafiki.constants.BudgetOption` and 
        ``<budget_amount>`` specifies the amount for the associated budget type.
        
        The following describes the budget types available:

        =====================       =====================
        **Budget Type**             **Description**
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
            BudgetOption.GPU_COUNT: 1,
            **budget
        }

        data = self._post('/train_jobs', json={
            'app': app,
            'task': task,
            'train_dataset_id': train_dataset_id,
            'val_dataset_id': val_dataset_id,
            'budget': budget,
            'model_ids': models
        })
        return data

    def get_train_jobs_by_user(self, user_id):
        '''
        Lists all of user's train jobs on Rafiki.

        :param str user_id: ID of the user
        :returns: Details of train jobs as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/train_jobs', params={ 
            'user_id': user_id
        })
        return data
    
    def get_train_jobs_of_app(self, app):
        '''
        Lists all of current user's train jobs associated to the app name on Rafiki.

        :param str app: Name of the app
        :returns: Details of train jobs as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/train_jobs/{}'.format(app))
        return data

    def get_train_job(self, app, app_version=-1):
        '''
        Retrieves details of the current user's train job identified by an app and an app version, 
        including workers' details.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        :returns: Details of train job as dictionary
        :rtype: dict[str, any]
        '''
        data = self._get('/train_jobs/{}/{}'.format(app, app_version))
        return data

    def get_best_trials_of_train_job(self, app, app_version=-1, max_count=2):
        '''
        Lists the best scoring trials of the current user's train job identified by an app and an app version,
        ordered by descending score.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        :param int max_count: Maximum number of trials to return
        :returns: Details of trials as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/train_jobs/{}/{}/trials'.format(app, app_version), params={
            'type': 'best',
            'max_count': max_count
        })
        return data

    def get_trials_of_train_job(self, app, app_version=-1):
        '''
        Lists all trials of the current user's train job identified by an app and an app version,
        ordered by when the trial started.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        :returns: Details of trials as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/train_jobs/{}/{}/trials'.format(app, app_version))
        return data

    def stop_train_job(self, app, app_version=-1):
        '''
        Prematurely stops the current user's train job identified by an app and an app version.
        Otherwise, the train job should stop by itself when its budget is reached.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        :returns: Stopped train job as dictionary
        :rtype: dict[str, any]
        '''
        data = self._post('/train_jobs/{}/{}/stop'.format(app, app_version))
        return data

    ####################################
    # Trials
    ####################################

    def get_trial(self, trial_id):
        '''
        Gets a specific trial.

        :param str trial_id: ID of trial
        :returns: Details of trial as dictionary
        :rtype: dict[str, any]
        '''
        data = self._get('/trials/{}'.format(trial_id))
        return data

    def get_trial_logs(self, trial_id):
        '''
        Gets the logs for a specific trial.

        :param str trial_id: ID of trial
        :returns: Logs of trial as dictionary
        :rtype: dict[str, any]
        '''
        data = self._get('/trials/{}/logs'.format(trial_id))
        return data

    # TODO: Fix method
    def get_trial_parameters(self, trial_id):
        '''
        Gets parameters of the model associated with the trial.

        :param str trial_id: ID of trial
        :returns: Parameters of the *trained* model associated with the trial
        :rtype: dict[str, any]
        '''
        data = self._get('/trials/{}/parameters'.format(trial_id))
        parameters = pickle.loads(data)
        return parameters

    def load_trial_model(self, trial_id, ModelClass):
        '''
        Loads an instance of a trial's model with the trial's knobs & parameters.

        Before this, you must have the trial's model class file already in your local filesystem,
        the dependencies of the model must have been installed separately, and the model class must have been 
        imported and passed into this method.

        Wraps :meth:`get_trial_parameters` and :meth:`get_trial`.

        :param str trial_id: ID of trial
        :param class ModelClass: model class that conincides with the trial's model class
        :returns: A *trained* model instance of ``ModelClass``, loaded with the trial's knobs and parameters
        '''
        data = self.get_trial(trial_id)
        knobs = data.get('knobs')
        parameters = self.get_trial_parameters(trial_id)
        model_inst = ModelClass(**knobs)
        model_inst.load_parameters(parameters)
        return model_inst

    ####################################
    # Inference Jobs
    ####################################

    def create_inference_job(self, app, app_version=-1):
        '''
        Creates and starts a inference job on Rafiki with the 2 best trials of an associated train job of the app. 
        The train job must have the status of ``STOPPED``.The inference job would be tagged with the train job's app and app version. 
        Throws an error if an inference job of the same train job is already running.

        In this method's response, `predictor_host` is this inference job's predictor's host. 

        Only admins, model developers & app developers can manage inference jobs. Model developers & app developers can only manage their own inference jobs.

        :param str app: Name of the app identifying the train job to use
        :param str app_version: Version of the app identifying the train job to use
        :returns: Created inference job as dictionary
        :rtype: dict[str, any]
        '''
        data = self._post('/inference_jobs', json={
            'app': app,
            'app_version': app_version
        })
        return data

    def get_inference_jobs_by_user(self, user_id):
        '''
        Lists all of user's inference jobs on Rafiki.

        :param str user_id: ID of the user
        :returns: Details of inference jobs as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/inference_jobs', params={ 
            'user_id': user_id
        })
        return data

    def get_inference_jobs_of_app(self, app):
        '''
        Lists all inference jobs associated to an app on Rafiki.

        :param str app: Name of the app
        :returns: Details of inference jobs as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/inference_jobs/{}'.format(app))
        return data

    def get_running_inference_job(self, app, app_version=-1):
        '''
        Retrieves details of the *running* inference job identified by an app and an app version,
        including workers' details.

        :param str app: Name of the app 
        :param int app_version: Version of the app (-1 for latest version)
        :returns: Details of inference job as dictionary
        :rtype: dict[str, any]
        '''
        data = self._get('/inference_jobs/{}/{}'.format(app, app_version))
        return data

    def stop_inference_job(self, app, app_version=-1):
        '''
        Stops the inference job identified by an app and an app version.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        :returns: Stopped inference job as dictionary
        :rtype: dict[str, any]
        '''
        data = self._post('/inference_jobs/{}/{}/stop'.format(app, app_version))
        return data

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