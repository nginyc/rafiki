import requests
import pprint

from common import BudgetType
from model import serialize_model

class Client(object):
    def __init__(self, admin_host='localhost', admin_port=8000):
        self._admin_host = admin_host
        self._admin_port = admin_port
        self._token = None

    def login(self, email, password):
        data = self._post('/tokens', form_data={
            'email': email,
            'password': password
        })
        self._token = data['token']

        # Abstract token from user
        del data['token']

        return data

    def logout(self):
        self._token = None


    ####################################
    # User
    ####################################

    def create_user(self, email, password, user_type):
        data = self._post('/users', form_data={
            'email': email,
            'password': password,
            'user_type': user_type
        })
        return data

    ####################################
    # Models
    ####################################

    def create_model(self, name, task, model_inst):
        model_serialized = serialize_model(model_inst)
        data = self._post(
            '/models', 
            files={
                'model_serialized': model_serialized
            },
            form_data={
                'name': name,
                'task': task
            }
        )
        return data

    def get_models(self):
        data = self._get('/models')
        return data

    def get_models_by_task(self, task):
        data = self._get('/models', params={
            'task': task
        })
        return data

    ####################################
    # Train Jobs
    ####################################
    
    def create_train_job(self, 
                        app_name, 
                        task, 
                        train_dataset_uri,
                        test_dataset_uri, 
                        budget_type=BudgetType.MODEL_TRIAL_COUNT, 
                        budget_amount=10):

        data = self._post('/train_jobs', form_data={
            'app_name': app_name,
            'task': task,
            'train_dataset_uri': train_dataset_uri,
            'test_dataset_uri': test_dataset_uri,
            'budget_type': budget_type,
            'budget_amount': budget_amount
        })
        return data
    
    def get_train_jobs(self, app_name):
        data = self._get('/train_jobs', params={
            'app_name': app_name
        })
        return data

    # Additionally returns a train job's models & workers' details
    def get_train_job(self, train_job_id):
        data = self._get('/train_jobs/{}'.format(train_job_id))
        return data

    def stop_train_job(self, train_job_id):
        data = self._post('/train_jobs/{}/stop'.format(train_job_id))
        return data

    ####################################
    # Train Job Workers
    ####################################

    # Only for train job workers
    def stop_train_job_worker(self, worker_id):
        data = self._post('/train_job_workers/{}/stop'.format(worker_id))
        return data

    ####################################
    # Trials
    ####################################
    
    # Returns only completed trials ordered by highest scores
    def get_best_trials_by_app(self, app_name, max_count=3):
        data = self._get('/trials', params={
            'app_name': app_name,
            'type': 'best',
            'max_count': max_count
        })
        return data

    # Returns all trials ordered from most recently started
    def get_trials_by_app(self, app_name):
        data = self._get('/trials', params={
            'app_name': app_name
        })
        return data

    ####################################
    # Utils
    ####################################

    def _get(self, path, params={}):
        url = 'http://{}:{}{}'.format(self._admin_host, self._admin_port, path)
        headers = self._get_headers()
        res = requests.get(
            url,
            headers=headers,
            params=params
        )
        return self._parse_response(res)

    def _post(self, path, params={}, files={}, form_data={}):
        url = 'http://{}:{}{}'.format(self._admin_host, self._admin_port, path)
        headers = self._get_headers()
        res = requests.post(
            url, 
            headers=headers,
            params=params, 
            data=form_data,
            files=files
        )
        return self._parse_response(res)

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