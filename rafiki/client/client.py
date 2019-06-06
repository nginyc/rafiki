import requests
import json
import pprint
import pickle
import os

from rafiki.constants import BudgetType, ModelAccessRight

class RafikiConnectionError(ConnectionError):
    pass

class Client(object):

    '''
    Initializes the Client to connect to a running 
    Rafiki Admin instance that the Client connects to.

    :param str admin_host: Host of Rafiki Admin
    :param int admin_port: Port of Rafiki Admin
    :param str advisor_host: Host of Rafiki Advisor
    :param int advisor_port: Port of Rafiki Advisor
    '''
    def __init__(self, admin_host='localhost', admin_port=3000,
                advisor_host='localhost', advisor_port=3002):
        self._admin_host = admin_host
        self._admin_port = admin_port
        self._advisor_host = advisor_host
        self._advisor_port = advisor_port
        self._token = None
        self._user = None

    def login(self, email, password):
        '''
        Creates a login session as a Rafiki user. You will have to be logged in to perform any actions.

        App developers can create, list and stop train and inference jobs, as well as list models.
        Model developers can create and list models.

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

    def create_users_with_csv(self, csv_file_path):
        '''
        Creates multiple Rafiki users with a CSV file.
        If user creation fails for a row (e.g. because the user's email already exists), the row will be skipped.

        :param str csv_file_path: Path to a single csv file containing users to seed

        :returns: Created users as list of dictionaries. 
        :rtype: dict[str, any][]
        '''

        f = open(csv_file_path, 'rb')
        csv_file_bytes = f.read()
        f.close()

        data = self._post(
            '/users/csv', 
            files={
                'csv_file_bytes': csv_file_bytes
            }
        )
        return data

    def get_users(self):
        '''
        Lists all Rafiki users.
        
        Only admins can list all users.

        :returns: List of users
        :rtype: dict[str, any][]
        '''
        data = self._get('/users')
        return data

    def delete_user(self, email):
        '''
        Deletes a Rafiki user. 
        
        Only admins can delete users (except for admins).
        Only superadmins can delete admins.

        :param str email: The user's email

        :returns: Deleted user as dictionary
        :rtype: dict[str, any]
        '''
        data = self._delete('/users', json={
            'email': email
        })
        return data

    ####################################
    # Models
    ####################################

    def create_model(self, name, task, model_file_path, model_class, docker_image=None, \
                    dependencies={}, access_right=ModelAccessRight.PRIVATE):
        '''
        Creates a model on Rafiki.

        Only admins & model developers can manage models.

        :param str name: Name of the model, must be unique on Rafiki
        :param str task: Task associated with the model, where the model must adhere to the specification of the task
        :param str model_file_path: Path to a single Python file that contains the definition for the model class
        :param obj model_class: The name of the model class inside the Python file. This class should implement :class:`rafiki.model.BaseModel`
        :param dependencies: List of dependencies & their versions
        :type dependencies: dict[str, str]
        :param str docker_image: A custom Docker image name that extends ``rafikiai/rafiki_worker``
        :param access_right: Model access right
        :type access_right: :class:`rafiki.constants.ModelAccessRight`
        :returns: Created model as dictionary
        :rtype: dict[str, any]

        ``model_file_path`` should point to a file that contains all necessary Python code for the model's implementation. 
        If the Python file imports any external Python modules, you should list it in ``dependencies`` or create a custom
        ``docker_image``. 

        If a model's ``access_right`` is set to ``PUBLIC``, all other users have access to the model for training
        and inference. By default, a model's access is ``PRIVATE``.

        ``dependencies`` should be a dictionary of ``{ <dependency_name>: <dependency_version> }``, where 
        ``<dependency_name>`` corresponds to the name of the Python Package Index (PyPI) package (e.g. ``tensorflow``)
        and ``<dependency_version>`` corresponds to the version of the PyPI package (e.g. ``1.12.0``). These dependencies 
        will be lazily installed on top of the worker's Docker image before the submitted model's code is executed.

        If the model is to be run on GPU, Rafiki could map dependencies to their GPU-enabled versions, if required. 
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

    def get_model(self, name):
        '''
        Retrieves details of a single model.

        :param str name: Name of model
        :returns: Details of model as dictionary
        :rtype: dict[str, any]
        '''
        data = self._get('/models/{}'.format(name))
        return data

    def get_models(self):
        '''
        Lists all models on Rafiki.

        :param str access_right: Model access right.
        :returns: Details of models as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/models')
        return data

    def get_models_of_task(self, task):
        '''
        Lists all models associated to a task on Rafiki.

        :param str task: Task name
        :returns: Details of models as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/models', params={
            'task': task
        })
        return data

    def download_model_file(self, name, model_file_path):
        '''
        Downloads the Python script containing the model's class to the local filesystem.

        :param str name: Name of model
        :param str model_file_path: Absolute/relative path to save the Python script to
        :returns: Details of model as dictionary
        :rtype: dict[str, any]
        '''
        model_file_bytes = self._get('/models/{}/model_file'.format(name))

        with open(model_file_path, 'wb') as f:
            f.write(model_file_bytes)

        data = self.get_model(name)
        dependencies = data.get('dependencies')
        model_class = data.get('model_class')

        print('Model file downloaded to "{}"!'.format(os.path.join(os.getcwd(), model_file_path)))
        
        if dependencies:
            print('You\'ll need to install the following model dependencies locally: {}'.format(dependencies))

        print('From the file, import the model class `{}`.'.format(model_class))

        return data

    ####################################
    # Train Jobs
    ####################################
    
    def create_train_job(self, app, task, train_dataset_uri, test_dataset_uri, budget, models=None):
        '''
        Creates and starts a train job on Rafiki. 
        A train job is uniquely identified by its associated app and the app version (returned in output).
        
        Only admins, model developers and app developers can manage train jobs.

        :param str app: Name of the app associated with the train job
        :param str task: Task associated with the train job, 
            the train job will train models associated with the task
        :param str train_dataset_uri: URI of the train dataset in a format specified by the task
        :param str test_dataset_uri: URI of the test (development) dataset in a format specified by the task
        :param str budget: Budget for each model
        :param str[] models: List of model names to use for train job
        :returns: Created train job as dictionary
        :rtype: dict[str, any]

        If ``models`` is unspecified, all models accessible to the user for the specified task will be used.

        ``budget`` should be a dictionary of ``{ <budget_type>: <budget_amount> }``, where 
        ``<budget_type>`` is one of :class:`rafiki.constants.BudgetType` and 
        ``<budget_amount>`` specifies the amount for the associated budget type.
        
        The following describes the budget types available:

        =====================       =====================
        **Budget Type**             **Description**
        ---------------------       ---------------------        
        ``MODEL_TRIAL_COUNT``       Target number of trials to run
        ``ENABLE_GPU``              Whether model training should run on GPU (0 or 1), if supported
        =====================       =====================
        '''

        data = self._post('/train_jobs', json={
            'app': app,
            'task': task,
            'train_dataset_uri': train_dataset_uri,
            'test_dataset_uri': test_dataset_uri,
            'budget': budget,
            'models': models
        })
        return data

    def get_train_jobs_by_user(self, user_id):
        '''
        Lists all train jobs associated to an user on Rafiki.

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
        Lists all train jobs associated to an app on Rafiki.

        :param str app: Name of the app
        :returns: Details of train jobs as list of dictionaries
        :rtype: dict[str, any][]
        '''
        data = self._get('/train_jobs/{}'.format(app))
        return data

    def get_train_job(self, app, app_version=-1):
        '''
        Retrieves details of the train job identified by an app and an app version, 
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
        Lists the best scoring trials of the train job identified by an app and an app version,
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
        Lists all trials of the train job identified by an app and an app version,
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
        Prematurely stops the train job identified by an app and an app version.
        Otherwise, the train job should stop by itself when its budget is reached.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        :returns: Stopped train job as dictionary
        :rtype: dict[str, any]
        '''
        data = self._post('/train_jobs/{}/{}/stop'.format(app, app_version))
        return data

    # Rafiki-internal method
    def stop_train_job_worker(self, service_id):
        data = self._post('/train_job_workers/{}/stop'.format(service_id))
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

        Only admins & app developers can manage inference jobs.

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
        Lists all inference jobs associated to an user on Rafiki.

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
    # Advisors
    ####################################

    def create_advisor(self, knob_config_str, advisor_id=None):
        '''
        Creates a Rafiki advisor. If `advisor_id` is passed, it will create an advisor
        of that ID, or do nothing if an advisor of that ID has already been created.

        :param str knob_config_str: Serialized knob configuration for advisor session
        :param str advisor_id: ID of advisor to create
        :returns: Created advisor as dictionary
        :rtype: dict[str, any]
        '''
        data = self._post('/advisors', target='advisor',
                            json={
                                'advisor_id': advisor_id,
                                'knob_config_str': knob_config_str
                            })
        return data

    def generate_proposal(self, advisor_id):
        '''
        Generate a proposal of knobs from an advisor.

        :param str advisor_id: ID of target advisor
        :returns: Knobs as `dict[<knob_name>, <knob_value>]`
        :rtype: dict[str, any]
        '''
        data = self._post('/advisors/{}/propose'.format(advisor_id), target='advisor')
        return data

    def feedback_to_advisor(self, advisor_id, knobs, score):
        '''
        Feedbacks to the advisor on the score of a set of knobs.
        Additionally returns another proposal of knobs after ingesting feedback.

        :param str advisor_id: ID of target advisor
        :param str knobs: Knobs to give feedback on
        :param float score: Score of the knobs, the higher the number, the better the set of knobs
        :returns: Knobs as `dict[<knob_name>, <knob_value>]`
        :rtype: dict[str, any]
        '''
        data = self._post('/advisors/{}/feedback'.format(advisor_id), 
                        target='advisor', json={
                            'score': score,
                            'knobs': knobs
                        })
        return data

    def delete_advisor(self, advisor_id):
        '''
        Deletes a Rafiki advisor.

        :param str advisor_id: ID of target advisor
        :returns: Deleted advisor as dictionary
        :rtype: dict[str, any]
        '''
        data = self._delete('/advisors/{}'.format(advisor_id), target='advisor')
        return data

    ####################################
    # Administrative Actions
    ####################################

    def stop_all_jobs(self):
        '''
        Stops all train and inference jobs on Rafiki. 

        Only admins can call this.
        '''
        data = self._post('/actions/stop_all_jobs')
        return data

    ####################################
    # Private
    ####################################

    def _get(self, path, params={}, target='admin'):
        url = self._make_url(path, target=target)
        headers = self._get_headers()
        res = requests.get(
            url,
            headers=headers,
            params=params
        )
        return self._parse_response(res)

    def _post(self, path, params={}, files={}, form_data=None, json=None, target='admin'):
        url = self._make_url(path, target=target)
        headers = self._get_headers()
        res = requests.post(
            url, 
            headers=headers,
            params=params, 
            data=form_data,
            json=json,
            files=files
        )
        return self._parse_response(res)

    def _delete(self, path, params={}, files={}, form_data=None, json=None, target='admin'):
        url = self._make_url(path, target=target)
        headers = self._get_headers()
        res = requests.delete(
            url, 
            headers=headers,
            params=params, 
            data=form_data,
            json=json,
            files=files
        )
        return self._parse_response(res)

    def _make_url(self, path, target='admin'):
        if target == 'admin':
            url = 'http://{}:{}{}'.format(self._admin_host, self._admin_port, path)
        elif target == 'advisor':
            url = 'http://{}:{}{}'.format(self._advisor_host, self._advisor_port, path)
        else:
            raise RafikiConnectionError('Invalid URL target: {}'.format(target))

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