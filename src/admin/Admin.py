import numpy as np
import os
import logging

from db import Database
from model import unserialize_model, serialize_model
from common import TrainJobWorkerStatus, UserType

from .worker import compute_train_worker_replicas_for_models
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
    DEFAULT_MODEL_IMAGE = 'rafiki_model'

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
        
        train_job_id = None
        app_version = None

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

            # Deploy train workers
            self._deploy_train_job_workers(train_job)

        return {
            'id': train_job_id,
            'app_version': app_version
        }

    def stop_train_job(self, train_job_id):
        with self._db:
            # Stop all workers for train job
            workers = self._db.get_workers_of_train_job(train_job_id)
            for worker in workers:
                self._destroy_train_job_worker(worker)

        return {
            'id': train_job_id
        }
            
    def get_train_job(self, train_job_id):
        with self._db:
            train_job = self._db.get_train_job(train_job_id)
            workers = self._db.get_workers_of_train_job(train_job_id)
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

    ####################################
    # Train Job Workers
    ####################################

    def stop_train_job_worker(self, worker_id):
        with self._db:
            worker = self._db.get_train_job_worker(worker_id)
            self._destroy_train_job_worker(worker)
        
        return {
            'id': worker_id
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
    # Others
    ####################################

    def clear_all_data(self):
        with self._db:
            self._db.clear_all_data()

    ####################################
    # Private Methods
    ####################################

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

    def _deploy_train_job_workers(self, train_job):
        workers = self._db.get_workers_of_train_job(train_job.id)
        models = self._db.get_models_of_task(train_job.task)
        model_to_replicas = compute_train_worker_replicas_for_models(models)

        for (model, replicas) in model_to_replicas.items():
            worker = next((x for x in workers if x.model_id == model.id), None)
            image_name = model.docker_image

            worker = self._db.create_train_job_worker(train_job.id, model.id)
            self._db.commit()

            # Create corresponding service for newly created worker
            service_id = self._create_service_for_worker(worker.id, image_name, replicas)
            self._db.update_train_job_worker(worker, service_id=service_id, replicas=replicas)
            self._db.commit()

    def _destroy_train_job_worker(self, worker):
        if worker.service_id is not None:
            self._container_manager.destroy_service(worker.service_id)

        self._db.mark_train_job_worker_as_stopped(worker)
        self._db.commit()

        # If all workers for the train job has have stopped, stop train job as well
        workers = self._db.get_workers_of_train_job(worker.train_job_id)
        if next((
            x for x in workers 
            if x.status in [TrainJobWorkerStatus.RUNNING, TrainJobWorkerStatus.STARTED]
        ), None) is None:
            train_job = self._db.get_train_job(worker.train_job_id)
            self._db.mark_train_job_as_complete(train_job)


    def _create_service_for_worker(self, worker_id, image_name, replicas):
        service_name = 'rafiki_train_worker_{}'.format(worker_id)
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

        # Mount logs folder onto workers too
        logs_folder_path = os.environ['LOGS_FOLDER_PATH']
        mounts = {
            logs_folder_path: logs_folder_path
        }

        service_id = self._container_manager.create_service(
            service_name=service_name, 
            image_name=image_name, 
            replicas=replicas, 
            args=[worker_id], 
            environment_vars=environment_vars,
            mounts=mounts
        )
        return service_id
        