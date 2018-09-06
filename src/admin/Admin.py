import numpy as np

from db import Database
from model import unserialize_model, serialize_model

from .auth import hash_password, if_hash_matches_password

class NoSuchUserException(Exception): 
    pass

class InvalidPasswordException(Exception):
    pass

class Admin(object):
    DEFAULT_MODEL_IMAGE_NAME = 'rafiki_worker'

    def __init__(self, db=Database()):
        self._db = db

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
            user = self._db.create_user(email, password_hash, user_type)
            self._db.commit()
            return {
                'id': user.id
            }

    ####################################
    # Train Job
    ####################################

    def create_train_job(self, user_id, app_name,
        task, train_dataset_uri, test_dataset_uri,
        budget_type, budget_amount):
        
        with self._db:
            # Compute auto-incremented app version
            train_jobs = self._db.get_train_jobs_by_app(app_name)
            app_version = max([x.app_version for x in train_jobs], default=0) + 1

            train_job = self._db.create_train_job(
                user_id=user_id,
                app_name=app_name,
                app_version=app_version,
                task=task,
                train_dataset_uri=train_dataset_uri,
                test_dataset_uri=test_dataset_uri,
                budget_type=budget_type,
                budget_amount=budget_amount
            )
            self._db.commit()
            
            return {
                'id': train_job.id,
                'app_version': train_job.app_version
            }

    def get_train_job(self, train_job_id):
        with self._db:
            train_job = self._db.get_train_job(train_job_id)
            workers = self._db.get_train_job_workers_by_train_job(train_job_id)
            models = self._db.get_models_by_task(train_job.task)
            return [
                {
                    'id': train_job.id,
                    'status': train_job.status,
                    'app_name': train_job.app_name,
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
            

    def get_train_jobs(self, app_name):
        with self._db:
            train_jobs = self._db.get_train_jobs_by_app(app_name)
            return [
                {
                    'id': x.id,
                    'status': x.status,
                    'app_name': x.app_name,
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
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app_name, max_models=3):
        with self._db:
            best_trials = self._db.get_best_trials_by_app(app_name, max_count=max_models)
            best_trials_models = [self._db.get_model(x.model_id) for x in best_trials]

            inference_job = self._db.create_inference_job(user_id, app_name)
            self._db.commit()

            # TODO: Deploy workers based on current deployment jobs

            return {
                'id': inference_job.id
            }

    def get_inference_jobs(self, app_name):
        with self._db:
            inference_jobs = self._db.get_inference_jobs_by_app(app_name)
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

    def get_best_trials_by_app(self, app_name, max_count=3):
        with self._db:
            best_trials = self._db.get_best_trials_by_app(app_name, max_count=max_count)
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

    def get_trials_by_app(self, app_name):
        with self._db:
            trials = self._db.get_trials_by_app(app_name)
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
            
        
    def get_trials_by_train_job(self, train_job_id):
        with self._db:
            trials = self._db.get_trials_by_train_job(train_job_id)
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
        model_serialized, docker_image_name=None):
        with self._db:
            model = self._db.create_model(
                user_id=user_id,
                name=name,
                task=task,
                model_serialized=model_serialized,
                docker_image_name=(docker_image_name or self.DEFAULT_MODEL_IMAGE_NAME)
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
                    'docker_image_name': model.docker_image_name
                }
                for model in models
            ]

    def get_models_by_task(self, task):
        with self._db:
            models = self._db.get_models_by_task(task)
            return [
                {
                    'name': model.name,
                    'task': model.task,
                    'datetime_created': model.datetime_created,
                    'user_id': model.user_id,
                    'docker_image_name': model.docker_image_name
                }
                for model in models
            ]

    ####################################
    # Others
    ####################################

    def clear_all_data(self):
        with self._db:
            self._db.clear_all_data()

