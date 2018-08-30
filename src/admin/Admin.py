import numpy as np
from db import Database, DatabaseConfig

from common import unserialize_model, serialize_model
from .auth import hash_password, if_hash_matches_password

class NoSuchUserException(Exception): 
    pass

class InvalidPasswordException(Exception):
    pass

class Admin(object):
    def __init__(self, database_config=DatabaseConfig()):
        self._db = Database(database_config)

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
    # Apps
    ####################################

    def create_app(self, user_id, name, task, train_dataset_uri, test_dataset_uri):
        # TODO: Validate that name is url-friendly

        with self._db:
            app = self._db.create_app(user_id, name, task, train_dataset_uri, test_dataset_uri)
            return {
                'name': app.name,
            }

    def get_app(self, name):
        with self._db:
            app = self._db.get_app_by_name(name)
            return {
                'name': app.name,
                'task': app.task,
                'datetime_created': app.datetime_created,
                'train_dataset_uri': app.train_dataset_uri,
                'test_dataset_uri': app.test_dataset_uri,
                'user_id': app.user_id
            }

    def get_apps(self):
        with self._db:
            apps = self._db.get_apps()
            return [
                {
                    'name': x.name,
                    'task': x.task,
                    'datetime_created': x.datetime_created
                }
                for x in apps
            ]

    ####################################
    # Train Job
    ####################################

    def create_train_job(self, user_id, app_name, budget_type, budget_amount):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            train_job = self._db.create_train_job(
                user_id=user_id,
                budget_type=budget_type,
                budget_amount=budget_amount,
                app_id=app.id
            )
            self._db.commit()

            # TODO: Deploy workers based on current train jobs
            
            return {
                'id': train_job.id
            }


    def get_train_jobs(self, app_name):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            train_jobs = self._db.get_train_jobs_by_app(app.id)
            return [
                {
                    'id': x.id,
                    'status': x.status,
                    'datetime_started': x.datetime_started,
                    'datetime_completed': x.datetime_completed,
                    'budget_type': x.budget_type,
                    'budget_amount': x.budget_amount
                }
                for x in train_jobs
            ]

    ####################################
    # Deployment Job
    ####################################

    def create_deployment_job(self, user_id, app_name, max_models=3):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            best_trials = self._db.get_best_trials_by_app(app.id, max_count=max_models)
            best_trials_models = [self._db.get_model(x.model_id) for x in best_trials]

            deployment_job = self._db.create_deployment_job(user_id, app.id)
            self._db.commit()

            # TODO: Deploy workers based on current deployment jobs

            return {
                'id': deployment_job.id
            }

    def get_deployment_jobs(self, app_name):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            deployment_jobs = self._db.get_deployment_jobs_by_app(app.id)
            return [
                {
                    'id': x.id,
                    'status': x.status,
                    'datetime_started': x.datetime_started,
                    'datetime_stopped': x.datetime_stopped,
                }
                for x in deployment_jobs
            ]
    
    ####################################
    # Trials
    ####################################

    def get_best_trials_by_app(self, app_name, max_count=3):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            best_trials = self._db.get_best_trials_by_app(app.id, max_count=max_count)
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

    def get_trials(self, app_name, train_job_id):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            trials = self._db.get_trials_by_train_job(app.id, train_job_id)
            trials_models = [self._db.get_model(x.model_id) for x in trials]
            return [
                {
                    'id': trial.id,
                    'hyperparameters': trial.hyperparameters,
                    'datetime_started': trial.datetime_started,
                    'model_name': model.name,
                    'score': trial.score
                }
                for (trial, model) in zip(trials, trials_models)
            ]

    def predict_with_trial(self, app_name, trial_id, queries):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            trial = self._db.get_trial(app.id, trial_id)
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

    def create_model(self, user_id, name, task, model_serialized):
        with self._db:
            model = self._db.create_model(
                user_id=user_id,
                name=name,
                task=task,
                model_serialized=model_serialized
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
                    'user_id': model.user_id
                }
                for model in models
            ]

    ####################################
    # Others
    ####################################

    def clear_all_data(self):
        with self._db:
            self._db.clear_all_data()

