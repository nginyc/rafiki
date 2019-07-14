#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import os
import logging
import traceback
import bcrypt
import uuid

from rafiki.db import Database
from rafiki.constants import ServiceStatus, UserType, ServiceType, InferenceJobStatus, \
    TrainJobStatus, ModelAccessRight, BudgetType
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.model import ModelLogger
from rafiki.container import DockerSwarmContainerManager
from rafiki.data_store import Dataset, FileDataStore, DataStore

from .services_manager import ServicesManager

logger = logging.getLogger(__name__)

class UserExistsError(Exception): pass
class UserAlreadyBannedError(Exception): pass
class InvalidUserError(Exception): pass
class InvalidPasswordError(Exception): pass
class InvalidRunningInferenceJobError(Exception): pass
class InvalidModelError(Exception): pass
class InvalidTrainJobError(Exception): pass
class InvalidTrialError(Exception): pass
class RunningInferenceJobExistsError(Exception): pass
class NoModelsForTrainJobError(Exception): pass
class InvalidDatasetError(Exception): pass

class Admin(object):
    def __init__(self, db=None, container_manager=None, data_store=None):
        self._db = db or Database()
        data_folder_path = os.path.join(os.environ['WORKDIR_PATH'], os.environ['DATA_DIR_PATH'])
        self._data_store: DataStore = data_store or FileDataStore(data_folder_path)
        self._base_worker_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_WORKER'],
                                                os.environ['RAFIKI_VERSION'])
        container_manager = container_manager or DockerSwarmContainerManager()
        self._services_manager = ServicesManager(self._db, container_manager)

    def seed(self):
        with self._db:
            self._seed_superadmin()

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
            'email': user.email,
            'user_type': user.user_type,
            'banned_date': user.banned_date
        }

    def create_user(self, email, password, user_type):
        user = self._create_user(email, password, user_type)
        return {
            'id': user.id,
            'email': user.email,
            'user_type': user.user_type
        }

    def get_users(self):
        users = self._db.get_users()
        return [
            {
                'id': user.id,
                'email': user.email,
                'user_type': user.user_type,
                'banned_date': user.banned_date
            }
            for user in users
        ]
        
    def get_user_by_email(self, email):
        user = self._db.get_user_by_email(email)
        if user is None:
            return None

        return {
            'id': user.id,
            'email': user.email,
            'user_type': user.user_type,
            'banned_date': user.banned_date
        }

    def ban_user(self, email):
        user = self._db.get_user_by_email(email)
        if user is None:
            raise InvalidUserError()
        if user.banned_date is not None:
            raise UserAlreadyBannedError()

        self._db.ban_user(user)
        
        return {
            'id': user.id,
            'email': user.email,
            'user_type': user.user_type,
            'banned_date': user.banned_date
        }
    
    ####################################
    # Datasets
    ####################################

    def create_dataset(self, user_id, name, task, data_file_path):
        # Store dataset in data folder
        store_dataset = self._data_store.save(data_file_path)

        # Get metadata for dataset
        store_dataset_id = store_dataset.id
        size_bytes = store_dataset.size_bytes
        owner_id = user_id

        dataset = self._db.create_dataset(name, task, size_bytes, store_dataset_id, owner_id)
        self._db.commit()

        return {
            'id': dataset.id,
            'name': dataset.name,
            'task': dataset.task,
            'size_bytes': dataset.size_bytes
        }

    def get_dataset(self, dataset_id):
        dataset = self._db.get_dataset(dataset_id)
        if dataset is None:
            raise InvalidDatasetError()

        return {
            'id': dataset.id,
            'name': dataset.name,
            'task': dataset.task,
            'datetime_created': dataset.datetime_created,
            'size_bytes': dataset.size_bytes,
            'owner_id': dataset.owner_id
        }

    def get_datasets(self, user_id, task=None):
        datasets = self._db.get_datasets(user_id, task)
        return [
            {
                'id': x.id,
                'name': x.name,
                'task': x.task,
                'datetime_created': x.datetime_created,
                'size_bytes': x.size_bytes

            }
            for x in datasets
        ]

    ####################################
    # Train Job
    ####################################

    def create_train_job(self, user_id, app, task, train_dataset_id, 
                        val_dataset_id, budget, model_ids, train_args):
        
        # Ensure at least 1 model
        if len(model_ids) == 0:
            raise NoModelsForTrainJobError()

        # Compute auto-incremented app version
        train_jobs = self._db.get_train_jobs_by_app(user_id, app)
        app_version = max([x.app_version for x in train_jobs], default=0) + 1

        # Get models available to user
        avail_model_ids = [x.id for x in self._db.get_available_models(user_id, task)]
        
        # Warn if there are no models for task
        if len(avail_model_ids) == 0:
            raise InvalidModelError(f'No models are available for task "{task}"')

        # Ensure all specified models are available
        for model_id in model_ids:
            if model_id not in avail_model_ids:
                raise InvalidModelError(f'No model with ID "{model_id}" is available for task "{task}"')

        # Ensure that datasets are valid and of the correct task
        try:
            train_dataset = self._db.get_dataset(train_dataset_id)
            assert train_dataset is not None
            assert train_dataset.task == task
            val_dataset = self._db.get_dataset(val_dataset_id)
            assert val_dataset is not None
            assert val_dataset.task == task
        except AssertionError as e:
            raise InvalidDatasetError(e)

        train_job = self._db.create_train_job(
            user_id=user_id,
            app=app,
            app_version=app_version,
            task=task,
            budget=budget,
            train_dataset_id=train_dataset_id,
            val_dataset_id=val_dataset_id,
            train_args=train_args
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

    def stop_train_job(self, user_id, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        self._services_manager.stop_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }
            
    def get_train_job(self, user_id, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        workers = self._db.get_workers_of_train_job(train_job.id)
        services = [self._db.get_service(x.service_id) for x in workers]
        worker_models = [self._db.get_model(self._db.get_sub_train_job(x.sub_train_job_id).model_id) \
                         for x in workers]

        return {
            'id': train_job.id,
            'status': train_job.status,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'task': train_job.task,
            'train_dataset_id': train_job.train_dataset_id,
            'val_dataset_id': train_job.val_dataset_id,
            'train_args': train_job.train_args,
            'datetime_started': train_job.datetime_started,
            'datetime_stopped': train_job.datetime_stopped,
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

    def get_train_jobs_by_app(self, user_id, app):
        train_jobs = self._db.get_train_jobs_by_app(user_id, app)
        return [
            {
                'id': x.id,
                'status': x.status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_id': x.train_dataset_id,
                'val_dataset_id': x.val_dataset_id,
                'train_args': x.train_args,
                'datetime_started': x.datetime_started,
                'datetime_stopped': x.datetime_stopped,
                'budget': x.budget
            }
            for x in train_jobs
        ]

    def get_best_trials_of_train_job(self, user_id, app, app_version=-1, max_count=2):
        train_job = self._db.get_train_job_by_app_version(user_id, app, app_version=app_version)
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
        return [
            {
                'id': x.id,
                'status': x.status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_id': x.train_dataset_id,
                'val_dataset_id': x.val_dataset_id,
                'train_args': x.train_args,
                'datetime_started': x.datetime_started,
                'datetime_stopped': x.datetime_stopped,
                'budget': x.budget
            }
            for x in train_jobs
        ]

    def get_trials_of_train_job(self, user_id, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(user_id, app, app_version=app_version)
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

    def stop_all_train_jobs(self):
        train_jobs = self._db.get_train_jobs_by_statuses([TrainJobStatus.STARTED, TrainJobStatus.RUNNING])
        for train_job in train_jobs:
            self._services_manager.stop_train_services(train_job.id)

        return [
            {
                'id': train_job.id
            }
            for train_job in train_jobs
        ]

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
            'worker_id': trial.worker_id
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

        with open(trial.params_file_path, 'rb') as f:
            parameters = f.read()

        return parameters

    ####################################
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app, app_version):
        train_job = self._db.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError('Have you started a train job for this app?')

        if train_job.status != TrainJobStatus.STOPPED:
            raise InvalidTrainJobError('Train job must be of status `STOPPED`.')

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

    def stop_inference_job(self, user_id, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(user_id, app, app_version=app_version)
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

    def get_running_inference_job(self, user_id, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(user_id, app, app_version=app_version)
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

    def get_inference_jobs_of_app(self, user_id, app):
        inference_jobs = self._db.get_inference_jobs_of_app(user_id, app)
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

    def stop_all_inference_jobs(self):
        inference_jobs = self._db.get_inference_jobs_by_status(InferenceJobStatus.RUNNING)
        for inference_job in inference_jobs:
            self._services_manager.stop_inference_services(inference_job.id)
            
        return [
            {
                'id': inference_job.id
            }
            for inference_job in inference_jobs
        ]

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, model_file_bytes, 
                    model_class, docker_image=None, dependencies={}, 
                    access_right=ModelAccessRight.PRIVATE):

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
        self._db.commit()

        return {
            'id': model.id,
            'user_id': model.user_id,
            'name': model.name 
        }

    def delete_model(self, model_id):
        model = self._db.get_model(model_id)
        if model is None:
            raise InvalidModelError()

        self._db.delete_model(model)
        
        return {
            'id': model.id,
            'user_id': model.user_id,
            'name': model.name 
        }

    def get_model_by_name(self, user_id, name):
        model = self._db.get_model_by_name(user_id, name)
        if model is None:
            raise InvalidModelError()

        return {
            'id': model.id,
            'user_id': model.user_id,
            'name': model.name,
            'task': model.task,
            'model_class': model.model_class,
            'datetime_created': model.datetime_created,
            'docker_image': model.docker_image,
            'dependencies': model.dependencies,
            'access_right': model.access_right
        }

    def get_model(self, model_id):
        model = self._db.get_model(model_id)
        if model is None:
            raise InvalidModelError()

        return {
            'id': model.id,
            'user_id': model.user_id,
            'name': model.name,
            'task': model.task,
            'model_class': model.model_class,
            'datetime_created': model.datetime_created,
            'docker_image': model.docker_image,
            'dependencies': model.dependencies,
            'access_right': model.access_right
        }

    def get_model_file(self, model_id):
        model = self._db.get_model(model_id)
        if model is None:
            raise InvalidModelError()

        return model.model_file_bytes

    def get_available_models(self, user_id, task=None):
        models = self._db.get_available_models(user_id, task)
        return [
            {
                'id': model.id,
                'user_id': model.user_id,
                'name': model.name,
                'task': model.task,
                'datetime_created': model.datetime_created,
                'dependencies': model.dependencies,
                'access_right': model.access_right
            }
            for model in models
        ]

    ####################################
    # Events
    ####################################

    def handle_event(self, name, **params):
        event_to_method = {
            'sub_train_job_budget_reached': self._on_sub_train_job_budget_reached,
            'train_job_worker_started': self._on_train_job_worker_started,
            'train_job_worker_stopped': self._on_train_job_worker_stopped
        }

        if name in event_to_method:
            event_to_method[name](**params)
        else:
            logger.error('Unknown event: "{}"'.format(name))

    def _on_sub_train_job_budget_reached(self, sub_train_job_id):
        self._services_manager.stop_sub_train_job_services(sub_train_job_id)

    def _on_train_job_worker_started(self, sub_train_job_id):
        sub_train_job = self._db.get_sub_train_job(sub_train_job_id)
        self._services_manager.refresh_train_job_status(sub_train_job.train_job_id)

    def _on_train_job_worker_stopped(self, sub_train_job_id):
        sub_train_job = self._db.get_sub_train_job(sub_train_job_id)
        self._services_manager.refresh_train_job_status(sub_train_job.train_job_id)
    
    ####################################
    # Private / Users
    ####################################

    def _seed_superadmin(self):
        logger.info('Seeding superadmin...')

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
        
