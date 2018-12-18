import os
import logging
import traceback
import bcrypt
import uuid
import csv

from rafiki.db import Database
from rafiki.constants import ServiceStatus, UserType, ServiceType, TrainJobStatus, ModelAccessRight, BudgetType
from rafiki.config import MIN_SERVICE_PORT, MAX_SERVICE_PORT, SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.model import ModelLogger
from rafiki.container import DockerSwarmContainerManager 

from .services_manager import ServicesManager

logger = logging.getLogger(__name__)

class UserExistsError(Exception): pass
class InvalidUserError(Exception): pass
class InvalidPasswordError(Exception): pass
class InvalidRunningInferenceJobError(Exception): pass
class InvalidModelError(Exception): pass
class InvalidModelAccessError(Exception): pass
class InvalidTrainJobError(Exception): pass
class InvalidTrialError(Exception): pass
class RunningInferenceJobExistsError(Exception): pass
class NoModelsForTrainJobError(Exception): pass

class Admin(object):
    def __init__(self, db=None, container_manager=None):
        if db is None: 
            db = Database()
        if container_manager is None: 
            container_manager = DockerSwarmContainerManager()
            
        self._base_worker_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_WORKER'],
                                                os.environ['RAFIKI_VERSION'])

        self._db = db
        self._services_manager = ServicesManager(db, container_manager)

    def seed(self):
        with self._db:
            self._seed_users()

    ####################################
    # Users
    ####################################

    def authenticate_user(self, email, password):
        user = self._db.get_user_by_email(email)

        if not user: 
            raise InvalidUserError()
        
        if not self._if_hash_matches_password(password, user.password_hash):
            raise InvalidPasswordError()

        return {
            'id': user.id,
            'user_type': user.user_type
        }

    def create_user(self, email, password, user_type):
        user = self._create_user(email, password, user_type)
        return {
            'id': user.id
        }

    def create_users(self, csv_file_bytes):
        temp_csv_file = '{}.csv'.format(str(uuid.uuid4()))

        # Temporarily save the csv file to disk
        with open(temp_csv_file, 'wb') as f:
            f.write(csv_file_bytes)

        users = []
        with open(temp_csv_file, 'rt', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.lower() for name in reader.fieldnames]
            line_number = 0
            for row in reader:
                if line_number > 0:
                    user = self._create_user(row['email'], row['password'], row['user_type'])
                    users.append(user)
                line_number += 1
        os.remove(temp_csv_file)
        return [
            {
                'id': user.id,
                'email': user.email,
                'user_type': user.user_type
            }
            for user in users
        ]

    ####################################
    # Train Job
    ####################################

    def create_train_job(self, user_id, app, task, train_dataset_uri, 
                        test_dataset_uri, budget, models=None):
        
        # Compute auto-incremented app version
        train_jobs = self._db.get_train_jobs_of_app(app)
        app_version = max([x.app_version for x in train_jobs], default=0) + 1

        # Get models available to user
        avail_models = self._db.get_models_of_task(user_id, task)
        
        # Auto populate models with all available models if not specified
        if models is None:
            model_ids = [x.id for x in avail_models]
        else:
            # Ensure all models are available
            model_ids = []
            for model in models:
                db_model = next((x for x in avail_models if x.name == model), None)
                if db_model is None:
                    raise InvalidModelAccessError('You don\'t have access to model "{}"'.format(model))
                model_ids.append(db_model.id)

        # Ensure that models are specified
        if len(model_ids) == 0:
            raise NoModelsForTrainJobError()

        train_job = self._db.create_train_job(
            user_id=user_id,
            app=app,
            app_version=app_version,
            task=task,
            budget=budget,
            train_dataset_uri=train_dataset_uri,
            test_dataset_uri=test_dataset_uri
        )
        self._db.commit()

        for model_id in model_ids:
            self._db.create_sub_train_job(
                train_job_id=train_job.id,
                model_id=model_id,
                user_id=train_job.user_id
            )

        train_job = self._services_manager.create_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def stop_train_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        self._services_manager.stop_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }
            
    def get_train_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        (status, datetime_started, datetime_stopped) = self._get_train_job_status(train_job)
        workers = self._get_workers_of_train_job(train_job)
        services = [self._db.get_service(x.service_id) for x in workers]
        worker_models = [self._db.get_model(self._db.get_sub_train_job(x.sub_train_job_id).model_id) \
                         for x in workers]

        return {
            'id': train_job.id,
            'status': status,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'task': train_job.task,
            'train_dataset_uri': train_job.train_dataset_uri,
            'test_dataset_uri': train_job.test_dataset_uri,
            'datetime_started': datetime_started,
            'datetime_stopped': datetime_stopped,
            'workers': [
                {
                    'service_id': service.id,
                    'status': service.status,
                    'replicas': service.replicas,
                    'datetime_started': service.datetime_started,
                    'datetime_stopped': service.datetime_stopped,
                    'model_name': model.name
                }
                for (worker, service, model) 
                in zip(workers, services, worker_models)
            ]
        }

    def get_train_jobs_of_app(self, app):
        train_jobs = self._db.get_train_jobs_of_app(app)
        statuses = [self._get_train_job_status(x) for x in train_jobs]
        return [
            {
                'id': x.id,
                'status': status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_uri': x.train_dataset_uri,
                'test_dataset_uri': x.test_dataset_uri,
                'datetime_started': datetime_started,
                'datetime_stopped': datetime_stopped,
                'budget': x.budget
            }
            for (x, (status, datetime_started, datetime_stopped)) in zip(train_jobs, statuses)
        ]

    def get_best_trials_of_train_job(self, app, app_version=-1, max_count=2):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        best_trials = self._db.get_best_trials_of_train_job(train_job.id, max_count=max_count)
        trials_models = [self._db.get_model(x.model_id) for x in best_trials]

        return [
            {
                'id': trial.id,
                'knobs': trial.knobs,
                'datetime_started': trial.datetime_started,
                'datetime_stopped': trial.datetime_stopped,
                'model_name': model.name,
                'score': trial.score
            }
            for (trial, model) in zip(best_trials, trials_models)
        ]

    def get_train_jobs_by_user(self, user_id):
        train_jobs = self._db.get_train_jobs_by_user(user_id)
        statuses = [self._get_train_job_status(x) for x in train_jobs]

        return [
            {
                'id': x.id,
                'status': status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_uri': x.train_dataset_uri,
                'test_dataset_uri': x.test_dataset_uri,
                'datetime_started': datetime_started,
                'datetime_stopped': datetime_stopped,
                'budget': x.budget
            }
            for (x, (status, datetime_started, datetime_stopped)) in zip(train_jobs, statuses)
        ]

    def get_trials_of_train_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        sub_train_jobs = self._db.get_sub_train_jobs_of_train_job(train_job.id)
        trials = []
        for sub_train_job in sub_train_jobs:
            trials_for_sub = self._db.get_trials_of_sub_train_job(sub_train_job.id)
            trials.extend(trials_for_sub)

        trials_models = [self._db.get_model(x.model_id) for x in trials]
        
        return [
            {
                'id': trial.id,
                'knobs': trial.knobs,
                'datetime_started': trial.datetime_started,
                'status': trial.status,
                'datetime_stopped': trial.datetime_stopped,
                'model_name': model.name,
                'score': trial.score
            }
            for (trial, model) in zip(trials, trials_models)
        ]

    def stop_train_job_worker(self, service_id):
        worker = self._services_manager.stop_train_job_worker(service_id)
        return {
            'service_id': worker.service_id,
            'train_job_id': worker.train_job_id,
            'sub_train_job_id': worker.sub_train_job_id
        }

    ####################################
    # Trials
    ####################################
    
    def get_trial(self, trial_id):
        trial = self._db.get_trial(trial_id)
        model = self._db.get_model(trial.model_id)
        
        if trial is None:
            raise InvalidTrialError()

        return {
            'id': trial.id,
            'knobs': trial.knobs,
            'datetime_started': trial.datetime_started,
            'status': trial.status,
            'datetime_stopped': trial.datetime_stopped,
            'model_name': model.name,
            'score': trial.score,
            'knobs': trial.knobs
        }

    def get_trial_logs(self, trial_id):
        trial = self._db.get_trial(trial_id)
        if trial is None:
            raise InvalidTrialError()

        trial_logs = self._db.get_trial_logs(trial_id)
        log_lines = [x.line for x in trial_logs]
        (messages, metrics, plots) = ModelLogger.parse_logs(log_lines)
        
        return {
            'plots': plots,
            'metrics': metrics,
            'messages': messages
        }

    def get_trial_parameters(self, trial_id):
        trial = self._db.get_trial(trial_id)
        if trial is None:
            raise InvalidTrialError()

        return trial.parameters

    ####################################
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app, app_version):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError('Have you started a train job for this app?')

        (status, _, _) = self._get_train_job_status(train_job)

        if status != TrainJobStatus.STOPPED:
            raise InvalidTrainJobError('Train job has not stopped.')

        # Ensure only 1 running inference job for 1 train job
        inference_job = self._db.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is not None:
            raise RunningInferenceJobExistsError()

        inference_job = self._db.create_inference_job(
            user_id=user_id,
            train_job_id=train_job.id
        )
        self._db.commit()

        (inference_job, predictor_service) = \
            self._services_manager.create_inference_services(inference_job.id)

        return {
            'id': inference_job.id,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'predictor_host': self._get_service_host(predictor_service)
        }

    def stop_inference_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._db.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._services_manager.stop_inference_services(inference_job.id)
        return {
            'id': inference_job.id,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def get_running_inference_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._db.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is None:
            raise InvalidRunningInferenceJobError()
            
        workers = self._db.get_workers_of_inference_job(inference_job.id)
        services = [self._db.get_service(x.service_id) for x in workers]
        predictor_service = self._db.get_service(inference_job.predictor_service_id)
        predictor_host = self._get_service_host(predictor_service)
        worker_trials = [self._db.get_trial(x.trial_id) for x in workers]
        worker_trial_models = [self._db.get_model(x.model_id) for x in worker_trials]

        return {
            'id': inference_job.id,
            'status': inference_job.status,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'datetime_started': inference_job.datetime_started,
            'datetime_stopped': inference_job.datetime_stopped,
            'predictor_host': predictor_host,
            'workers': [
                {
                    'service_id': service.id,
                    'status': service.status,
                    'replicas': service.replicas,
                    'datetime_started': service.datetime_started,
                    'datetime_stopped': service.datetime_stopped,
                    'trial': {
                        'id': trial.id,
                        'score': trial.score,
                        'knobs': trial.knobs,
                        'model_name': model.name
                    }
                }
                for (worker, service, trial, model) 
                in zip(workers, services, worker_trials, worker_trial_models)
            ]
        }

    def get_inference_jobs_of_app(self, app):
        inference_jobs = self._db.get_inference_jobs_of_app(app)
        train_jobs = [self._db.get_train_job(x.train_job_id) for x in inference_jobs]
        predictor_services = [self._db.get_service(x.predictor_service_id) for x in inference_jobs]
        predictor_hosts = [self._get_service_host(x) for x in predictor_services]
        return [
            {
                'id': inference_job.id,
                'status': inference_job.status,
                'train_job_id': train_job.id,
                'app': train_job.app,
                'app_version': train_job.app_version,
                'datetime_started': inference_job.datetime_started,
                'datetime_stopped': inference_job.datetime_stopped,
                'predictor_host': predictor_host
            }
            for (inference_job, train_job, predictor_host) in zip(inference_jobs, train_jobs, predictor_hosts)
        ]

    def get_inference_jobs_by_user(self, user_id):
        inference_jobs = self._db.get_inference_jobs_by_user(user_id)
        train_jobs = [self._db.get_train_job(x.train_job_id) for x in inference_jobs]
        predictor_services = [self._db.get_service(x.predictor_service_id) for x in inference_jobs]
        predictor_hosts = [self._get_service_host(x) for x in predictor_services]
        return [
            {
                'id': inference_job.id,
                'status': inference_job.status,
                'train_job_id': train_job.id,
                'app': train_job.app,
                'app_version': train_job.app_version,
                'datetime_started': inference_job.datetime_started,
                'datetime_stopped': inference_job.datetime_stopped,
                'predictor_host': predictor_host
            }
            for (inference_job, train_job, predictor_host) in zip(inference_jobs, train_jobs, predictor_hosts)
        ]

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, model_file_bytes, 
                    model_class, docker_image=None, dependencies={}, access_right=ModelAccessRight.PRIVATE):
        
        model = self._db.create_model(
            user_id=user_id,
            name=name,
            task=task,
            model_file_bytes=model_file_bytes,
            model_class=model_class,
            docker_image=(docker_image or self._base_worker_image),
            dependencies=dependencies,
            access_right=access_right
        )

        return {
            'name': model.name 
        }

    def get_model(self, user_id, name):
        model = self._db.get_model_by_name(name)
        if model is None:
            raise InvalidModelError()

        if model.access_right == ModelAccessRight.PRIVATE and model.user_id != user_id:
            raise InvalidModelAccessError()

        return {
            'name': model.name,
            'task': model.task,
            'model_class': model.model_class,
            'datetime_created': model.datetime_created,
            'user_id': model.user_id,
            'docker_image': model.docker_image,
            'dependencies': model.dependencies,
            'access_right': model.access_right
        }

    def get_model_file(self, user_id, name):
        model = self._db.get_model_by_name(name)
        
        if model is None:
            raise InvalidModelError()

        if model.access_right == ModelAccessRight.PRIVATE and model.user_id != user_id:
            raise InvalidModelAccessError()

        return model.model_file_bytes

    def get_models(self, user_id):
        models = self._db.get_models(user_id)
        return [
            {
                'name': model.name,
                'task': model.task,
                'model_class': model.model_class,
                'datetime_created': model.datetime_created,
                'user_id': model.user_id,
                'docker_image': model.docker_image,
                'dependencies': model.dependencies,
                'access_right': model.access_right
            }
            for model in models
        ]

    def get_models_of_task(self, user_id, task):
        models = self._db.get_models_of_task(user_id, task)
        return [
            {
                'name': model.name,
                'task': model.task,
                'model_class': model.model_class,
                'datetime_created': model.datetime_created,
                'user_id': model.user_id,
                'docker_image': model.docker_image,
                'dependencies': model.dependencies,
                'access_right': model.access_right
            }
            for model in models
        ]
        
    ####################################
    # Private / Users
    ####################################

    def _seed_users(self):
        logger.info('Seeding users...')

        # Seed superadmin
        try:
            self._create_user(
                email=SUPERADMIN_EMAIL,
                password=SUPERADMIN_PASSWORD,
                user_type=UserType.SUPERADMIN
            )
        except UserExistsError:
            logger.info('Skipping superadmin creation as it already exists...')

    def _hash_password(self, password):
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return password_hash

    def _if_hash_matches_password(self, password, password_hash):
        return bcrypt.checkpw(password.encode('utf-8'), password_hash)

    def _create_user(self, email, password, user_type):
        password_hash = self._hash_password(password)
        user = self._db.get_user_by_email(email)

        if user is not None:
            raise UserExistsError()

        user = self._db.create_user(email, password_hash, user_type)
        self._db.commit()
        return user

    ####################################
    # Private / Train Job
    ####################################

    # Returns (status, datetime_started, datetime_stopped)
    def _get_train_job_status(self, train_job):
        sub_train_jobs = self._db.get_sub_train_jobs_of_train_job(train_job.id)

        count = {
            TrainJobStatus.STARTED: 0,
            TrainJobStatus.RUNNING: 0,
            TrainJobStatus.STOPPED: 0
        }

        datetime_started = None
        datetime_stopped = None
        for sub_train_job in sub_train_jobs:
            count[sub_train_job.status] += 1
            if sub_train_job.datetime_started is not None and \
                (datetime_started is None or datetime_started < sub_train_job.datetime_started):
                datetime_started = sub_train_job.datetime_started

            if sub_train_job.datetime_stopped is not None and \
                (datetime_stopped is None or datetime_stopped > sub_train_job.datetime_stopped):
                datetime_stopped = sub_train_job.datetime_stopped

        # Determine status of train job based on sub-jobs
        if count[TrainJobStatus.STOPPED] == len(sub_train_jobs):
            return (TrainJobStatus.STOPPED, datetime_started, datetime_stopped)
        elif count[TrainJobStatus.STARTED] == len(sub_train_jobs):
            return (TrainJobStatus.STARTED, datetime_started, None)
        else:
            return (TrainJobStatus.RUNNING, datetime_started, None)

    def _get_workers_of_train_job(self, train_job):
        workers = []
        sub_train_jobs = self._db.get_sub_train_jobs_of_train_job(train_job.id)
        for sub_train_job in sub_train_jobs:
            workers += self._db.get_workers_of_sub_train_job(sub_train_job.id)
        return workers

    ####################################
    # Private / Services
    ####################################

    def _get_service_host(self, service):
        return '{}:{}'.format(service.ext_hostname, service.ext_port)

    ####################################
    # Private / Others
    ####################################

    def __enter__(self):
        self.connect()

    def connect(self):
        self._db.connect()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def disconnect(self):
        self._db.disconnect()
        
