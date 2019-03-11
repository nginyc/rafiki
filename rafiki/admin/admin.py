import os
import logging
import traceback
import bcrypt
import uuid
import csv

from rafiki.meta_store import MetaStore
from rafiki.constants import ServiceStatus, UserType, ServiceType, TrainJobStatus, \
                            ModelAccessRight, BudgetType, InferenceJobStatus
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.model import LoggerUtils
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
    def __init__(self, meta_store=None, container_manager=None):
        if meta_store is None: 
            meta_store = MetaStore()
            
        if container_manager is None: 
            container_manager = DockerSwarmContainerManager()
            
        self._base_worker_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_WORKER'],
                                                os.environ['RAFIKI_VERSION'])

        self._meta_store = meta_store
        self._services_manager = ServicesManager(meta_store, container_manager)

    def seed(self):
        with self._meta_store:
            self._seed_users()

    ####################################
    # Users
    ####################################

    def authenticate_user(self, email, password):
        user = self._meta_store.get_user_by_email(email)

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
            for row in reader:
                user = self._create_user(row['email'], row['password'], row['user_type'])
                users.append(user)
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
                        val_dataset_uri, budget, models=None):
        
        train_jobs = self._meta_store.get_train_jobs_of_app(app)

        # Ensure there is no existing train job for app
        if any([x.status in [TrainJobStatus.RUNNING, TrainJobStatus.STARTED] for x in train_jobs]):
            raise InvalidTrainJobError('Another train job for app "{}" is still running!'.format(app))
        
        # Compute auto-incremented app version
        app_version = max([x.app_version for x in train_jobs], default=0) + 1

        # Get models available to user
        avail_models = self._meta_store.get_models_of_task(user_id, task)
        
        if models is None:
            # Auto populate models with all available models if not specified
            model_id_to_config = { db_model.id: {} for db_model in avail_models }
        else:
            model_id_to_config = {}
            # Ensure all models are available
            for (name, config) in models.items():
                db_model = next((x for x in avail_models if x.name == name), None)
                if db_model is None:
                    raise InvalidModelAccessError('You don\'t have access to model "{}"'.format(name))

                model_id_to_config[db_model.id] = config

        # Ensure that models are specified
        if len(model_id_to_config) == 0:
            raise NoModelsForTrainJobError()

        # Create train & sub train jobs in DB
        train_job = self._meta_store.create_train_job(
            user_id=user_id,
            app=app,
            app_version=app_version,
            task=task,
            budget=budget,
            train_dataset_uri=train_dataset_uri,
            val_dataset_uri=val_dataset_uri
        )
        self._meta_store.commit()

        for (model_id, config) in model_id_to_config.items():
            self._meta_store.create_sub_train_job(
                train_job_id=train_job.id,
                model_id=model_id,
                config=config
            )
        self._meta_store.commit()

        train_job = self._services_manager.create_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def stop_train_job(self, app, app_version=-1):
        train_job = self._meta_store.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        self._services_manager.stop_train_services(train_job.id)
        self.refresh_train_job_status(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }
            
    def get_train_job(self, app, app_version=-1):
        train_job = self._meta_store.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(train_job.id)
        services = [self._meta_store.get_service(x.service_id) for x in sub_train_jobs]
        models = [self._meta_store.get_model(x.model_id) for x in sub_train_jobs]

        return {
            'id': train_job.id,
            'status': train_job.status,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'task': train_job.task,
            'train_dataset_uri': train_job.train_dataset_uri,
            'val_dataset_uri': train_job.val_dataset_uri,
            'datetime_started': train_job.datetime_started,
            'datetime_stopped': train_job.datetime_stopped,
            'services': [
                {
                    'service_id': service.id,
                    'status': service.status,
                    'replicas': service.replicas,
                    'datetime_started': service.datetime_started,
                    'datetime_stopped': service.datetime_stopped,
                    'model_name': model.name
                }
                for (sub_train_job, service, model) 
                in zip(sub_train_jobs, services, models)
            ]
        }

    def get_train_jobs_of_app(self, app):
        train_jobs = self._meta_store.get_train_jobs_of_app(app)
        return [
            {
                'id': x.id,
                'status': x.status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_uri': x.train_dataset_uri,
                'val_dataset_uri': x.val_dataset_uri,
                'datetime_started': x.datetime_started,
                'datetime_stopped': x.datetime_stopped,
                'budget': x.budget
            }
            for x in train_jobs
        ]

    def get_best_trials_of_train_job(self, app, app_version=-1, max_count=2):
        train_job = self._meta_store.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        best_trials = self._meta_store.get_best_trials_of_train_job(train_job.id, max_count=max_count)
        trials_models = [self._meta_store.get_model(x.model_id) for x in best_trials]

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
        train_jobs = self._meta_store.get_train_jobs_by_user(user_id)

        return [
            {
                'id': x.id,
                'status': x.status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_uri': x.train_dataset_uri,
                'val_dataset_uri': x.val_dataset_uri,
                'datetime_started': x.datetime_started,
                'datetime_stopped': x.datetime_stopped,
                'budget': x.budget
            }
            for x in train_jobs
        ]

    def get_trials_of_train_job(self, app, app_version=-1):
        train_job = self._meta_store.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        trials = self._meta_store.get_trials_of_train_job(train_job.id)
        trials_models = [self._meta_store.get_model(x.model_id) for x in trials]
        
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

    def stop_sub_train_job(self, sub_train_job_id):
        sub_train_job = self._services_manager.stop_sub_train_job_services(sub_train_job_id)
        return {
            'id': sub_train_job.id,
            'service_id': sub_train_job.service_id,
            'train_job_id': sub_train_job.train_job_id,
        }

    def stop_all_train_jobs(self):
        train_jobs = self._meta_store.get_train_jobs_by_statuses(
            [TrainJobStatus.STARTED, TrainJobStatus.RUNNING, TrainJobStatus.ERRORED]
        )
        for train_job in train_jobs:
            self._services_manager.stop_train_services(train_job.id)
            self.refresh_train_job_status(train_job.id)

        return [
            {
                'id': train_job.id
            }
            for train_job in train_jobs
        ]

    def refresh_train_job_status(self, train_job_id):
        train_job = self._meta_store.get_train_job(train_job_id)
        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(train_job.id)
        services = [self._meta_store.get_service(x.service_id) for x in sub_train_jobs]

        count = {
            ServiceStatus.STARTED: 0,
            ServiceStatus.DEPLOYING: 0,
            ServiceStatus.RUNNING: 0,
            ServiceStatus.ERRORED: 0,
            ServiceStatus.STOPPED: 0
        }

        for service in services:
            if service is None:
                continue
            count[service.status] += 1

        # Determine status of train job based on sub-jobs
        if count[ServiceStatus.ERRORED] > 0:
            self._meta_store.mark_train_job_as_errored(train_job)
        elif count[ServiceStatus.STOPPED] == len(sub_train_jobs):
            self._meta_store.mark_train_job_as_stopped(train_job)
        elif count[ServiceStatus.RUNNING] > 0:
            self._meta_store.mark_train_job_as_running(train_job)

    ####################################
    # Trials
    ####################################
    
    def get_trial(self, trial_id):
        trial = self._meta_store.get_trial(trial_id)
        model = self._meta_store.get_model(trial.model_id)
        
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
        trial = self._meta_store.get_trial(trial_id)
        if trial is None:
            raise InvalidTrialError()

        trial_logs = self._meta_store.get_trial_logs(trial_id)
        log_lines = [x.line for x in trial_logs]
        (messages, metrics, plots) = LoggerUtils.parse_logs(log_lines)
        
        return {
            'plots': plots,
            'metrics': metrics,
            'messages': messages
        }

    def get_trial_parameters(self, trial_id):
        trial = self._meta_store.get_trial(trial_id)
        if trial is None:
            raise InvalidTrialError()

        # TODO: Fix
        return None

    ####################################
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app, app_version):
        train_job = self._meta_store.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError('Have you started a train job for this app?')

        if train_job.status != TrainJobStatus.STOPPED:
            raise InvalidTrainJobError('Train job must be of status "STOPPED".')

        # Ensure only 1 running inference job for 1 train job
        inference_job = self._meta_store.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is not None:
            raise RunningInferenceJobExistsError()

        # Get trials to load for inference job
        best_trials = self._meta_store.get_best_trials_of_train_job(train_job.id, max_count=2)
        if len(best_trials) == 0:
            raise InvalidTrainJobError('Train job has no trials with saved models!')

        # Create inference & sub inference jobs in DB
        train_job = self._meta_store.create_inference_job(
            user_id=user_id,
            train_job_id=train_job.id
        )
        self._meta_store.commit()
        for trial in best_trials:
            self._meta_store.create_sub_inference_job(
                inference_job_id=inference_job.id,
                trial_id=trial.id,
            )
        self._meta_store.commit()

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
        train_job = self._meta_store.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._meta_store.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._services_manager.stop_inference_services(inference_job.id)
        self.refresh_inference_job_status(inference_job.id)

        return {
            'id': inference_job.id,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def get_running_inference_job(self, app, app_version=-1):
        train_job = self._meta_store.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._meta_store.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is None:
            raise InvalidRunningInferenceJobError()
        
        sub_inference_jobs = self._meta_store.get_sub_inference_jobs_of_inference_job(inference_job.id)
        services = [self._meta_store.get_service(x.service_id) for x in sub_inference_jobs]
        trials = [self._meta_store.get_trial(x.trial_id) for x in sub_inference_jobs]
        models = [self._meta_store.get_model(x.model_id) for x in trials]
        predictor_service = self._meta_store.get_service(inference_job.predictor_service_id)
        predictor_host = self._get_service_host(predictor_service)

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
                for (sub_inference_job, service, trial, model) 
                in zip(sub_inference_jobs, services, trials, models)
            ]
        }

    def get_inference_jobs_of_app(self, app):
        inference_jobs = self._meta_store.get_inference_jobs_of_app(app)
        train_jobs = [self._meta_store.get_train_job(x.train_job_id) for x in inference_jobs]
        predictor_services = [self._meta_store.get_service(x.predictor_service_id) for x in inference_jobs]
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
        inference_jobs = self._meta_store.get_inference_jobs_by_user(user_id)
        train_jobs = [self._meta_store.get_train_job(x.train_job_id) for x in inference_jobs]
        predictor_services = [self._meta_store.get_service(x.predictor_service_id) for x in inference_jobs]
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

    def stop_all_inference_jobs(self):
        inference_jobs = self._meta_store.get_inference_jobs_by_statuses(
            [InferenceJobStatus.STARTED, InferenceJobStatus.RUNNING, InferenceJobStatus.ERRORED]
        )
        for inference_job in inference_jobs:
            self._services_manager.stop_inference_services(inference_job.id)
            self.refresh_inference_job_status(inference_job.id)
            
        return [
            {
                'id': inference_job.id
            }
            for inference_job in inference_jobs
        ]

    def refresh_inference_job_status(self, inference_job_id):
        inference_job = self._meta_store.get_inference_job(inference_job_id)
        sub_inference_jobs = self._meta_store.get_sub_inference_jobs_of_inference_job(inference_job.id)
        services = [self._meta_store.get_service(x.service_id) for x in sub_inference_jobs]
        predictor_service = self._meta_store.get_service(inference_job.predictor_service_id)

        count = {
            ServiceStatus.STARTED: 0,
            ServiceStatus.DEPLOYING: 0,
            ServiceStatus.RUNNING: 0,
            ServiceStatus.ERRORED: 0,
            ServiceStatus.STOPPED: 0
        }

        for service in services:
            if service is None:
                continue
            count[service.status] += 1

        # If predictor is errored or any sub inference jobs is errored, errored
        if predictor_service is None or predictor_service.status == ServiceStatus.ERRORED or \
           count[ServiceStatus.ERRORED] > 0:
           self._meta_store.mark_inference_job_as_errored(inference_job)

        # If all inference jobs are either stopped, stopped
        elif count[ServiceStatus.STOPPED] == len(sub_inference_jobs):
            self._meta_store.mark_inference_job_as_stopped(inference_job)
        
        # If predictor is running and at least 1 sub inference job is running, running
        if predictor_service.status == ServiceStatus.RUNNING or \
            count[ServiceStatus.RUNNING] > 0:
            self._meta_store.mark_inference_job_as_running(inference_job)

    ####################################
    # Events
    ####################################

    def handle_event(self, name, **params):
        event_to_method = {
            'sub_train_job_budget_reached': self._on_sub_train_job_budget_reached,
            'sub_train_job_worker_started': self._on_sub_train_job_worker_started,
            'sub_train_job_worker_stopped': self._on_sub_train_job_worker_stopped,
            'sub_inference_job_worker_started': self._on_sub_inference_job_worker_started,
            'sub_inference_job_worker_stopped': self._on_sub_inference_job_worker_stopped,
            'inference_job_predictor_started': self._on_sub_inference_job_worker_stopped
        }

        if name in event_to_method:
            event_to_method[name](**params)
        else:
            logger.log('Unknown event: "{}"'.format(name))

    def _on_sub_train_job_budget_reached(self, sub_train_job_id):
        self.stop_sub_train_job(sub_train_job_id)

    def _on_sub_train_job_worker_started(self, sub_train_job_id):
        sub_train_job = self._meta_store.get_sub_train_job(sub_train_job_id)
        self.refresh_train_job_status(sub_train_job.train_job_id)

    def _on_sub_train_job_worker_stopped(self, sub_train_job_id):
        sub_train_job = self._meta_store.get_sub_train_job(sub_train_job_id)
        self.refresh_train_job_status(sub_train_job.train_job_id)
    
    def _on_sub_inference_job_worker_started(self, sub_inference_job_id):
        sub_inference_job = self._meta_store.get_sub_train_job(sub_inference_job_id)
        self.refresh_inference_job_status(sub_inference_job.inference_job_id)

    def _on_sub_inference_job_worker_stopped(self, sub_inference_job_id):
        sub_inference_job = self._meta_store.get_sub_train_job(sub_inference_job_id)
        self.refresh_inference_job_status(sub_inference_job.inference_job_id)

    def _on_inference_job_predictor_started(self, inference_job_id):
        self.refresh_inference_job_status(inference_job_id)

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, model_file_bytes, 
                    model_class, docker_image=None, dependencies={}, access_right=ModelAccessRight.PRIVATE):
        
        model = self._meta_store.create_model(
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
        model = self._meta_store.get_model_by_name(name)
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
        model = self._meta_store.get_model_by_name(name)
        
        if model is None:
            raise InvalidModelError()

        if model.access_right == ModelAccessRight.PRIVATE and model.user_id != user_id:
            raise InvalidModelAccessError()

        return model.model_file_bytes

    def get_models(self, user_id):
        models = self._meta_store.get_models(user_id)
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
        models = self._meta_store.get_models_of_task(user_id, task)
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
        user = self._meta_store.get_user_by_email(email)

        if user is not None:
            raise UserExistsError()

        user = self._meta_store.create_user(email, password_hash, user_type)
        self._meta_store.commit()
        return user

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
        self._meta_store.connect()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def disconnect(self):
        self._meta_store.disconnect()
        
