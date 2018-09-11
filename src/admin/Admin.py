import numpy as np
import os
import logging
import traceback

from db import Database
from model import unserialize_model, serialize_model
from common import ServiceStatus, UserType, ServiceType

from .train import compute_train_worker_replicas_for_models
from .containers import DockerSwarmContainerManager 
from .auth import hash_password, if_hash_matches_password

logger = logging.getLogger(__name__)

class UserExistsException(Exception):
    pass

class NoSuchUserException(Exception): 
    pass

class InvalidPasswordException(Exception):
    pass

class Admin(object):
    DEFAULT_MODEL_IMAGE = 'rafiki_worker'

    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._container_manager = container_manager
        self._superadmin_email = os.environ['SUPERADMIN_EMAIL']
        self._superadmin_password = os.environ['SUPERADMIN_PASSWORD']
        self._seed_users()

    ####################################
    # Users
    ####################################

    def authenticate_user(self, email, password):
        with self._db:
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
        password_hash = hash_password(password)
        with self._db:
            user = self._db.get_user_by_email(email)

            if user is not None:
                raise UserExistsException()

            user = self._db.create_user(email, password_hash, user_type)
            self._db.commit()
            return {
                'id': user.id
            }

    ####################################
    # Train Job
    ####################################

    def create_train_job(self, user_id, app,
        task, train_dataset_uri, test_dataset_uri,
        budget_type, budget_amount):
        
        with self._db:
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

            train_job_id = train_job.id
            app_version = train_job.app_version

            self._deploy_train_job_services(train_job)

        return {
            'id': train_job_id,
            'app_version': app_version
        }

    def stop_train_job(self, train_job_id):
        with self._db:
            # Stop all services for train job
            train_job_services = self._db.get_services_of_train_job(train_job_id)
            for train_job_service in train_job_services:
                self._destroy_train_job_service(train_job_service)

        return {
            'id': train_job_id
        }
            
    def get_train_job(self, train_job_id):
        with self._db:
            train_job = self._db.get_train_job(train_job_id)
            workers = self._db.get_services_of_train_job(train_job_id)
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
                            'model': next(y.name for y in models if y.id == x.model_id),
                            'service_id': x.service_id,
                            'datetime_started': x.datetime_started,
                            'status': x.status,
                            'replicas': x.replicas
                        }
                        for x in workers
                    ],
                    'models': [x.name for x in models]
                }
                
            ]

    def get_train_jobs_of_app(self, app):
        with self._db:
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

    def stop_train_job_service(self, service_id):
        with self._db:
            train_job_service = self._db.get_train_job_service(service_id)
            self._destroy_train_job_service(train_job_service)
        
            return {
                'service_id': train_job_service.service_id,
                'model_id': train_job_service.model_id,
                'train_job_id': train_job_service.train_job_id
            }

    ####################################
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app, max_models=3):
        with self._db:
            best_trials = self._db.get_best_trials_of_app(app, max_count=max_models)
            best_trials_models = [self._db.get_model(x.model_id) for x in best_trials]

            inference_job = self._db.create_inference_job(user_id, app)
            self._db.commit()

            # TODO: Deploy workers based on current deployment jobs

            return {
                'id': inference_job.id
            }

    def get_inference_jobs(self, app):
        with self._db:
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
        with self._db:
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
        with self._db:
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
        with self._db:
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

    def predict_with_trial(self, trial_id, queries):
        with self._db:
            trial = self._db.get_trial(trial_id)
            model = self._db.get_model(trial.model_id)
            
            # Load model based on trial & make predictions
            model_inst = unserialize_model(model.model_serialized)
            model_inst.init(trial.hyperparameters)
            model_inst.load_parameters(trial.parameters)
            preds = model_inst.predict(queries)
            model_inst.destroy()

            return preds
            
    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, 
        model_serialized, docker_image=None):
        with self._db:
            model = self._db.create_model(
                user_id=user_id,
                name=name,
                task=task,
                model_serialized=model_serialized,
                docker_image=(docker_image or self.DEFAULT_MODEL_IMAGE)
            )

            return {
                'name': model.name 
            }

    def get_models(self):
        with self._db:
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
        with self._db:
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
    # Services
    ####################################

    def _deploy_train_job_services(self, train_job):
        models = self._db.get_models_of_task(train_job.task)
        model_to_replicas = compute_train_worker_replicas_for_models(models)

        for (model, replicas) in model_to_replicas.items():
            # Create corresponding service for newly created worker
            self._create_train_job_service(train_job, model, replicas)

        self._update_train_job_status(train_job.id)

    def _create_train_job_service(self, train_job, model, replicas):
        service_type = ServiceType.TRAIN
        environment_vars = {
            'POSTGRES_HOST': os.environ['POSTGRES_HOST'],
            'POSTGRES_PORT': os.environ['POSTGRES_PORT'],
            'POSTGRES_USER': os.environ['POSTGRES_USER'],
            'POSTGRES_DB': os.environ['POSTGRES_DB'],
            'POSTGRES_PASSWORD': os.environ['POSTGRES_PASSWORD'],
            'LOGS_FOLDER_PATH': os.environ['LOGS_FOLDER_PATH'],
            'ADMIN_HOST': os.environ['ADMIN_HOST'],
            'ADMIN_PORT': os.environ['ADMIN_PORT'],
            'SUPERADMIN_EMAIL': self._superadmin_email,
            'SUPERADMIN_PASSWORD': self._superadmin_password
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            replicas=replicas,
            environment_vars=environment_vars
        )

        self._db.create_train_job_service(
            service_id=service.id,
            train_job_id=train_job.id,
            model_id=model.id
        )

    def _destroy_train_job_service(self, train_job_service):
        train_job_id = train_job_service.train_job_id
        service_id = train_job_service.service_id

        service = self._db.get_service(service_id)
        self._destroy_service(service)

        self._update_train_job_status(train_job_id)

    def _update_train_job_status(self, train_job_id):
        train_job_services = self._db.get_services_of_train_job(train_job_id)
        services = [self._db.get_service(x.service_id) for x in train_job_services]
        
        # If all services for the train job have stopped, stop train job as well
        if next((
            x for x in services 
            if x.status in [ServiceStatus.RUNNING, ServiceStatus.STARTED]
        ), None) is None:
            train_job = self._db.get_train_job(train_job_id)
            self._db.mark_train_job_as_complete(train_job)

    def _destroy_service(self, service):
        if service.container_service_id is not None:
            self._container_manager.destroy_service(service.container_service_id)

        self._db.mark_service_as_stopped(service)

        # Mark its workers as stopped
        workers = self._db.get_workers_of_service(service.id)
        for worker in workers:
            self._db.mark_worker_as_stopped(worker)

        self._db.commit()

    def _create_service(self, service_type, docker_image,
                        replicas, environment_vars):
        # Mount logs folder onto workers too
        logs_folder_path = os.environ['LOGS_FOLDER_PATH']
        mounts = {
            logs_folder_path: logs_folder_path
        }
        container_manager_type = type(self._container_manager).__name__

        service = self._db.create_service(
            container_manager_type=container_manager_type,
            service_type=service_type,
            docker_image=docker_image
        )
        self._db.commit()

        try:
            container_service_name = 'rafiki_service_{}'.format(service.id)
            container_service_id = self._container_manager.create_service(
                service_name=container_service_name,
                docker_image=docker_image, 
                replicas=replicas, 
                args=[service.id], # Pass service id to workers
                environment_vars=environment_vars,
                mounts=mounts
            )

            self._db.mark_service_as_running(
                service,
                container_service_name=container_service_name,
                container_service_id=container_service_id,
                replicas=replicas
            )
            self._db.commit()

        except Exception:
            logger.error('Error while creating service with ID {}'.format(service.id))
            logger.error(traceback.format_exc())
            self._db.mark_service_as_errored(service)
            self._db.commit()

        return service
        
    ####################################
    # Others
    ####################################

    def clear_all_data(self):
        with self._db:
            self._db.clear_all_data()
    
    def _seed_users(self):
        logger.info('Seeding users...')

        # Seed superadmin
        try:
            self.create_user(
                email=self._superadmin_email,
                password=self._superadmin_password,
                user_type=UserType.SUPERADMIN
            )
        except UserExistsException:
            logger.info('Skipping superadmin creation as it already exists...')
