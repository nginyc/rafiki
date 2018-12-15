import requests
import json
import pprint

from rafiki.constants import BudgetType, ModelAccessRight

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

        :returns: Dictionary of shape `{ id, user_type }`, or `None` if client is not logged in
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
        
        Only admins can manage users.

        :param str email: The new user's email
        :param str password: The new user's password
        :param user_type: The new user's type
        :type user_type: :class:`rafiki.constants.UserType` 
        '''
        data = self._post('/users', json={
            'email': email,
            'password': password,
            'user_type': user_type
        })
        return data

    ####################################
    # Models
    ####################################

    def create_model(self, name, task, model_file_path, model_class, docker_image=None, \
                    dependencies={}, access_right=ModelAccessRight.PUBLIC):
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
        :param str access_right: Model access right
        
        ``model_file_path`` should point to a file that contains all necessary Python code for the model's implementation. 
        If the Python file imports any external Python modules, you should list it in ``dependencies`` or create a custom
        ``docker_image``.

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

    def get_models(self):
        '''
        Lists all models on Rafiki.

        :param str access_right: Model access right.
        '''
        data = self._get('/models')
        return data

    def get_models_of_task(self, task):
        '''
        Lists all models associated to a task on Rafiki.

        :param str task: Task name
        '''
        data = self._get('/models', params={
            'task': task
        })
        return data

    ####################################
    # Train Jobs
    ####################################
    
    def create_train_job(self, 
                        app, 
                        task, 
                        train_dataset_uri,
                        test_dataset_uri, 
                        budget=None):
        '''
        Creates and starts a train job on Rafiki. 
        A train job is uniquely identified by its associated app and the app version (returned in output).
        
        Only admins & app developers can manage train jobs.

        :param str app: Name of the app associated with the train job
        :param str task: Task associated with the train job, 
            the train job will train models associated with the task
        :param str train_dataset_uri: URI of the train dataset in a format specified by the task
        :param str test_dataset_uri: URI of the test (development) dataset in a format specified by the task
        :param budget: budget for the train job
        :type budget: dict[:class:`rafiki.constants.BudgetType`, int]

        ``budget`` should be a dictionary of ``{ <budget_type>: <budget_amount> }``, where 
        ``<budget_type>`` is one of :class:`rafiki.constants.BudgetType` and 
        ``<budget_amount>`` specifies the amount for the associated budget type.
        
        The following describes the budget types available:

        =====================       =====================
        **Budget Type**             **Description**
        ---------------------       ---------------------        
        ``MODEL_TRIAL_COUNT``       Target number of trials, per model, to run
        ``ENABLE_GPU``              Whether model training should run on GPU (0 or 1), if supported
        =====================       =====================
        '''

        data = self._post('/train_jobs', json={
            'app': app,
            'task': task,
            'train_dataset_uri': train_dataset_uri,
            'test_dataset_uri': test_dataset_uri,
            'budget': budget
        })
        return data

    def get_train_jobs_by_user(self, user_id):
        '''
        Lists all train jobs associated to an user on Rafiki.

        :param str user_id: ID of the user
        '''
        data = self._get('/train_jobs', params={ 
            'user_id': user_id
        })
        return data
    
    def get_train_jobs_of_app(self, app):
        '''
        Lists all train jobs associated to an app on Rafiki.

        :param str app: Name of the app
        '''
        data = self._get('/train_jobs/{}'.format(app))
        return data

    def get_train_job(self, app, app_version=-1):
        '''
        Retrieves details of the train job identified by an app and an app version, 
        including workers' details.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        '''
        data = self._get('/train_jobs/{}/{}'.format(app, app_version))
        return data

    def get_best_trials_of_train_job(self, app, app_version=-1, max_count=3):
        '''
        Lists the best scoring trials of the train job identified by an app and an app version,
        ordered by descending score.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        :param int max_count: Maximum number of trials to return
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
        '''
        data = self._get('/train_jobs/{}/{}/trials'.format(app, app_version))
        return data

    def stop_train_job(self, app, app_version=-1):
        '''
        Prematurely stops the train job identified by an app and an app version.
        Otherwise, the train job should stop by itself when its budget is reached.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
        '''
        data = self._post('/train_jobs/{}/{}/stop'.format(app, app_version))
        return data

    def stop_train_job_worker(self, service_id):
        '''
        Rafiki-internal method
        '''
        data = self._post('/train_job_workers/{}/stop'.format(service_id))
        return data

    ####################################
    # Trials
    ####################################

    def get_trial(self, trial_id):
        '''
        Gets a specific trial.

        :param str trial_id: ID of trial
        '''
        data = self._get('/trials/{}'.format(trial_id))
        return data

    def get_trial_logs(self, trial_id):
        '''
        Gets the logs for a specific trial.

        :param str trial_id: ID of trial
        '''
        data = self._get('/trials/{}/logs'.format(trial_id))
        return data

    ####################################
    # Inference Jobs
    ####################################

    def create_inference_job(self, app, app_version=-1):
        '''
        Creates and starts a inference job on Rafiki with the 2 best trials of an associated train job of the app. 
        The train job must have the status of ``COMPLETED``.The inference job would be tagged with the train job's app and app version. 
        Throws an error if an inference job of the same train job is already running.

        In this method's response, `predictor_host` is this inference job's predictor's host. 

        Only admins & app developers can manage inference jobs.

        :param str app: Name of the app identifying the train job to use
        :param str app_version: Version of the app identifying the train job to use
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
        '''
        data = self._get('/inference_jobs', params={ 
            'user_id': user_id
        })
        return data

    def get_inference_jobs_of_app(self, app):
        '''
        Lists all inference jobs associated to an app on Rafiki.

        :param str app: Name of the app
        '''
        data = self._get('/inference_jobs/{}'.format(app))
        return data

    def get_running_inference_job(self, app, app_version=-1):
        '''
        Retrieves details of the *running* inference job identified by an app and an app version,
        including workers' details.

        :param str app: Name of the app 
        :param int app_version: Version of the app (-1 for latest version)
        '''
        data = self._get('/inference_jobs/{}/{}'.format(app, app_version))
        return data

    def stop_inference_job(self, app, app_version=-1):
        '''
        Stops the inference job identified by an app and an app version.

        :param str app: Name of the app
        :param int app_version: Version of the app (-1 for latest version)
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
        '''
        data = self._post('/advisors/{}/propose'.format(advisor_id), target='advisor')
        return data

    def feedback_to_advisor(self, advisor_id, knobs, score):
        '''
        Feedbacks to the advisor on the score of a set of knobs.
        Additionally returns another proposal of knobs after ingesting feedback.

        :param str advisor_id: ID of target advisor
        :param str knobs: Knobs to give feedback on
        :rtype: dict[<knob_name>, <knob_value>]
        :param float score: Score of the knobs, the higher the number, the better the set of knobs
        :returns: Knobs as `dict[<knob_name>, <knob_value>]`
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
        '''
        data = self._delete('/advisors/{}'.format(advisor_id), target='advisor')
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
            raise Exception('Invalid URL target: {}'.format(target))

        return url

    def _parse_response(self, res):
        if res.status_code != 200:
            raise Exception(res.text)

        data = res.json()
        return data

    def _get_headers(self):
        if self._token is not None:
            return {
                'Authorization': 'Bearer ' + self._token
            }
        else:
            return {}