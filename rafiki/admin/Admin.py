import numpy as np
import os
import logging
import traceback

from rafiki.db import Database
from rafiki.model import unserialize_model, serialize_model
from rafiki.constants import ServiceStatus, UserType, ServiceType
from rafiki.config import BASE_MODEL_IMAGE, QUERY_FRONTEND_IMAGE, \
    MIN_SERVICE_PORT, MAX_SERVICE_PORT, QUERY_FRONTEND_PORT

from .containers import DockerSwarmContainerManager 
from .auth import hash_password, if_hash_matches_password
from .ServicesManager import ServicesManager

logger = logging.getLogger(__name__)

class UserExistsException(Exception):
    pass

class NoSuchUserException(Exception): 
    pass

class InvalidPasswordException(Exception):
    pass

class Admin(object):
    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._services_manager = ServicesManager(db, container_manager)
        
        with self._db:
            self._seed_users()

    ####################################
    # Users
    ####################################

    def authenticate_user(self, email, password):
        user = self._db.get_user_by_email(email)

        if not user: 
            raise NoSuchUserException()
        
        if not if_hash_matches_password(password, user.password_hash):
            raise InvalidPasswordException()

        return {
            'id': user.id,
            'user_type': user.user_type
        }

    def create_user(self, email, password, user_type):
        user = self._create_user(email, password, user_type)
        return {
            'id': user.id
        }

    ####################################
    # Train Job
    ####################################

    def create_train_job(self, user_id, app,
        task, train_dataset_uri, test_dataset_uri,
        budget_type, budget_amount):
        
        # Compute auto-incremented app version
        train_jobs = self._db.get_train_jobs_of_app(app)
        app_version = max([x.app_version for x in train_jobs], default=0) + 1

        train_job = self._db.create_train_job(
            user_id=user_id,
            app=app,
            app_version=app_version,
            task=task,
            train_dataset_uri=train_dataset_uri,
            test_dataset_uri=test_dataset_uri,
            budget_type=budget_type,
            budget_amount=budget_amount
        )
        self._db.commit()

        train_job = self._services_manager.create_train_job_services(train_job.id)

        return {
            'id': train_job.id,
            'app_version': train_job.app_version
        }

    def stop_train_job(self, train_job_id):
        self._services_manager.stop_train_job_services(train_job_id)

        return {
            'id': train_job_id
        }
            
    def get_train_job(self, train_job_id):
        train_job = self._db.get_train_job(train_job_id)
        workers = self._db.get_workers_of_train_job(train_job_id)
        services = self._db.get_services(ids=[x.id for x in workers])
        models = self._db.get_models_of_task(train_job.task)
        return [
            {
                'id': train_job.id,
                'status': train_job.status,
                'app': train_job.app,
                'app_version': train_job.app_version,
                'task': train_job.task,
                'train_dataset_uri': train_job.train_dataset_uri,
                'test_dataset_uri': train_job.test_dataset_uri,
                'datetime_started': train_job.datetime_started,
                'datetime_completed': train_job.datetime_completed,
                'budget_type': train_job.budget_type,
                'budget_amount': train_job.budget_amount,
                'workers': [
                    {
                        'service_id': x.service_id,
                        'status': x.status,
                        'datetime_started': x.datetime_started,
                        'datetime_stopped': x.datetime_stopped,
                        'replicas': x.replicas
                    }
                    for x in services
                ],
                'models': [x.name for x in models]
            }
        ]

    def get_train_jobs_of_app(self, app):
        train_jobs = self._db.get_train_jobs_of_app(app)
        return [
            {
                'id': x.id,
                'status': x.status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_uri': x.train_dataset_uri,
                'test_dataset_uri': x.test_dataset_uri,
                'datetime_started': x.datetime_started,
                'datetime_completed': x.datetime_completed,
                'budget_type': x.budget_type,
                'budget_amount': x.budget_amount
            }
            for x in train_jobs
        ]

    def stop_train_job_worker(self, service_id):
        worker = self._services_manager.stop_train_job_worker(service_id)
        return {
            'service_id': worker.service_id,
            'model_id': worker.model_id,
            'train_job_id': worker.train_job_id
        }

    ####################################
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app, app_version):
        inference_job = self._db.create_inference_job(
            user_id=user_id,
            app=app,
            app_version=app_version
        )
        self._db.commit()

        (inference_job, query_service) = \
            self._services_manager.create_inference_job_services(inference_job.id)

        return {
            'id': inference_job.id,
            'query_host': '{}:{}'.format(query_service.ext_hostname, query_service.ext_port)
        }

    def stop_inference_job(self, inference_job_id):
        inference_job = self._services_manager.stop_inference_job_services(inference_job_id)
        return {
            'id': inference_job.id
        }

    def get_inference_jobs(self, app):
        inference_jobs = self._db.get_inference_jobs_of_app(app)
        return [
            {
                'id': x.id,
                'status': x.status,
                'datetime_started': x.datetime_started,
                'datetime_stopped': x.datetime_stopped,
            }
            for x in inference_jobs
        ]
    
    ####################################
    # Trials
    ####################################

    def get_best_trials_of_app(self, app, max_count=3):
        best_trials = self._db.get_best_trials_of_app(app, max_count=max_count)
        best_trials_models = [self._db.get_model(x.model_id) for x in best_trials]
        return [
            {
                'id': trial.id,
                'train_job_id': trial.train_job_id,
                'hyperparameters': trial.hyperparameters,
                'datetime_started': trial.datetime_started,
                'model_name': model.name,
                'score': trial.score
            }
            for (trial, model) in zip(best_trials, best_trials_models)
        ]

    def get_trials_of_app(self, app):
        trials = self._db.get_trials_of_app(app)
        trials_models = [self._db.get_model(x.model_id) for x in trials]
        return [
            {
                'id': trial.id,
                'status': trial.status,
                'train_job_id': trial.train_job_id,
                'hyperparameters': trial.hyperparameters,
                'datetime_started': trial.datetime_started,
                'datetime_completed': trial.datetime_completed,
                'model_name': model.name,
                'score': trial.score
            }
            for (trial, model) in zip(trials, trials_models)
        ] 
        
    def get_trials_of_train_job(self, train_job_id):
        trials = self._db.get_trials_of_train_job(train_job_id)
        trials_models = [self._db.get_model(x.model_id) for x in trials]
        return [
            {
                'id': trial.id,
                'train_job_id': trial.train_job_id,
                'hyperparameters': trial.hyperparameters,
                'datetime_started': trial.datetime_started,
                'model_name': model.name,
                'score': trial.score
            }
            for (trial, model) in zip(trials, trials_models)
        ]

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, 
        model_serialized, docker_image=None):
        model = self._db.create_model(
            user_id=user_id,
            name=name,
            task=task,
            model_serialized=model_serialized,
            docker_image=(docker_image or BASE_MODEL_IMAGE)
        )

        return {
            'name': model.name 
        }

    def get_models(self):
        models = self._db.get_models()
        return [
            {
                'name': model.name,
                'task': model.task,
                'datetime_created': model.datetime_created,
                'user_id': model.user_id,
                'docker_image': model.docker_image
            }
            for model in models
        ]

    def get_models_of_task(self, task):
        models = self._db.get_models_of_task(task)
        return [
            {
                'name': model.name,
                'task': model.task,
                'datetime_created': model.datetime_created,
                'user_id': model.user_id,
                'docker_image': model.docker_image
            }
            for model in models
        ]
        
    ####################################
    # Private
    ####################################

    def _seed_users(self):
        logger.info('Seeding users...')

        # Seed superadmin
        try:
            superadmin_email = os.environ['SUPERADMIN_EMAIL']
            superadmin_password = os.environ['SUPERADMIN_PASSWORD']
            self._create_user(
                email=superadmin_email,
                password=superadmin_password,
                user_type=UserType.SUPERADMIN
            )
        except UserExistsException:
            logger.info('Skipping superadmin creation as it already exists...')

    def _create_user(self, email, password, user_type):
        password_hash = hash_password(password)
        user = self._db.get_user_by_email(email)

        if user is not None:
            raise UserExistsException()

        user = self._db.create_user(email, password_hash, user_type)
        self._db.commit()
        return user

    def __enter__(self):
        self.connect()

    def connect(self):
        self._db.connect()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def disconnect(self):
        self._db.disconnect()