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
import bcrypt

from rafiki.constants import ServiceStatus, UserType, TrainJobStatus, ModelAccessRight, InferenceJobStatus
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.meta_store import MetaStore
from rafiki.model import LoggerUtils
from rafiki.container import DockerSwarmContainerManager 
from rafiki.container import KubernetesContainerManager
from rafiki.data_store import FileDataStore, DataStore
from rafiki.param_store import FileParamStore, ParamStore

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
    def __init__(self, meta_store=None, container_manager=None, data_store=None, param_store=None):
        self._meta_store = meta_store or MetaStore()
        container_manager = None
        if os.getenv('CONTAINER_MODE', 'SWARM') == 'SWARM':
            container_manager = container_manager or DockerSwarmContainerManager()
        else:
            container_manager = container_manager or KubernetesContainerManager()
        self._data_store: DataStore = data_store or FileDataStore()
        self._param_store: ParamStore = param_store or FileParamStore()
        self._base_worker_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_WORKER'],
                                                os.environ['RAFIKI_VERSION'])
        self._services_manager = ServicesManager(self._meta_store, container_manager)

    def seed(self):
        with self._meta_store:
            self._seed_superadmin()

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
        users = self._meta_store.get_users()
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
        user = self._meta_store.get_user_by_email(email)
        if user is None:
            return None

        return {
            'id': user.id,
            'email': user.email,
            'user_type': user.user_type,
            'banned_date': user.banned_date
        }

    def ban_user(self, email):
        user = self._meta_store.get_user_by_email(email) 
        if user is None:
            raise InvalidUserError()
        if user.banned_date is not None:
            raise UserAlreadyBannedError()

        self._meta_store.ban_user(user)
        
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

        dataset = self._meta_store.create_dataset(name, task, size_bytes, store_dataset_id, owner_id)
        self._meta_store.commit()

        return {
            'id': dataset.id,
            'name': dataset.name,
            'task': dataset.task,
            'size_bytes': dataset.size_bytes
        }

    def get_dataset(self, dataset_id):
        dataset = self._meta_store.get_dataset(dataset_id)
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
        datasets = self._meta_store.get_datasets(user_id, task)
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
                        val_dataset_id, budget, model_ids, train_args={}):
        
        # Ensure there is no existing train job for app
        train_jobs = self._meta_store.get_train_jobs_by_app(user_id, app)
        if any([x.status in [TrainJobStatus.RUNNING, TrainJobStatus.STARTED] for x in train_jobs]):
            raise InvalidTrainJobError('Another train job for app "{}" is still running!'.format(app))
        
        # Ensure at least 1 model
        if len(model_ids) == 0:
            raise NoModelsForTrainJobError()

        # Compute auto-incremented app version # config.load_kube_config(config_file='/root/rafiki/k8sconfig')
        logger.info('config k8s')
        # self._client_service = kubernetes.client.CoreV1Api()
        app_version = max([x.app_version for x in train_jobs], default=0) + 1

        # Get models available to user
        avail_model_ids = [x.id for x in self._meta_store.get_available_models(user_id, task)]

        # Warn if there are no models for task  
        if len(avail_model_ids) == 0:
            raise InvalidModelError(f'No models are available for task "{task}"')

        # Ensure all specified models are available
        for model_id in model_ids:
            if model_id not in avail_model_ids:
                raise InvalidModelError(f'No model with ID "{model_id}" is available for task "{task}"')

        # Ensure that datasets are valid and of the correct task
        try:
            train_dataset = self._meta_store.get_dataset(train_dataset_id)
            assert train_dataset is not None
            assert train_dataset.task == task
            val_dataset = self._meta_store.get_dataset(val_dataset_id)
            assert val_dataset is not None
            assert val_dataset.task == task
        except AssertionError as e:
            raise InvalidDatasetError(e)

        # Create train & sub train jobs in DB
        train_job = self._meta_store.create_train_job(
            user_id=user_id,
            app=app,
            app_version=app_version,
            task=task,
            budget=budget,
            train_dataset_id=train_dataset_id,
            val_dataset_id=val_dataset_id,
            train_args=train_args
        )
        self._meta_store.commit()

        for model_id in model_ids:
            self._meta_store.create_sub_train_job(
                train_job_id=train_job.id,
                model_id=model_id
            )

        self._meta_store.commit()

        self._services_manager.create_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def stop_train_job(self, user_id, app, app_version=-1):
        train_job = self._meta_store.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        self._services_manager.stop_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def stop_sub_train_job(self, sub_train_job_id):
        self._services_manager.stop_sub_train_job_services(sub_train_job_id)

        return {
            'id': sub_train_job_id
        }
            
    def get_train_job(self, user_id, app, app_version=-1):
        train_job = self._meta_store.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

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
            'datetime_stopped': train_job.datetime_stopped
        }

    def get_train_jobs_by_app(self, user_id, app):
        train_jobs = self._meta_store.get_train_jobs_by_app(user_id, app)
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

    def get_train_jobs_by_user(self, user_id):
        train_jobs = self._meta_store.get_train_jobs_by_user(user_id)
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
    def stop_all_train_jobs(self):
        train_jobs = self._meta_store.get_train_jobs_by_statuses(
                    [TrainJobStatus.STARTED, TrainJobStatus.RUNNING])
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
        trial = self._meta_store.get_trial(trial_id)
        model = self._meta_store.get_model(trial.model_id)
        
        if trial is None:
            raise InvalidTrialError()

        return {
            'id': trial.id,
            'no': trial.no,
            'worker_id': trial.worker_id,
            'proposal': trial.proposal,
            'datetime_started': trial.datetime_started,
            'status': trial.status,
            'datetime_stopped': trial.datetime_stopped,
            'model_name': model.name,
            'score': trial.score,
            'is_params_saved': trial.is_params_saved
        }

    def get_best_trials_of_train_job(self, user_id, app, app_version=-1, max_count=2):
        train_job = self._meta_store.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        best_trials = self._meta_store.get_best_trials_of_train_job(train_job.id, max_count=max_count)
        trials_models = [self._meta_store.get_model(x.model_id) for x in best_trials]

        return [
            {
                'id': trial.id,
                'proposal': trial.proposal,
                'datetime_started': trial.datetime_started,
                'status': trial.status,
                'datetime_stopped': trial.datetime_stopped,
                'model_name': model.name,
                'score': trial.score,
                'is_params_saved': trial.is_params_saved
            }
            for (trial, model) in zip(best_trials, trials_models)
        ]

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

        if not trial.is_params_saved:
            raise InvalidTrialError('Trial\'s model parameters were not saved')

        params = self._param_store.load(trial.store_params_id)
        return params

    def get_trials_of_train_job(self, user_id, app, app_version=-1, limit=1000, offset=0):
        train_job = self._meta_store.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError()

        trials = self._meta_store.get_trials_of_train_job(train_job.id, limit=limit, offset=offset)
        trials_models = [self._meta_store.get_model(x.model_id) for x in trials]
        
        return [
            {
                'id': trial.id,
                'no': trial.no,
                'worker_id': trial.worker_id,
                'proposal': trial.proposal,
                'datetime_started': trial.datetime_started,
                'status': trial.status,
                'datetime_stopped': trial.datetime_stopped,
                'model_name': model.name,
                'score': trial.score,
                'is_params_saved': trial.is_params_saved
            }
            for (trial, model) in zip(trials, trials_models)
        ]

    ####################################
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app, app_version, budget):
        train_job = self._meta_store.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobError('Have you started a train job for this app?')

        if train_job.status != TrainJobStatus.STOPPED:
            raise InvalidTrainJobError('Train job must be of status `STOPPED`.')

        # Ensure only 1 running inference job for 1 train job
        inference_job = self._meta_store.get_deployed_inference_job_by_train_job(train_job.id)
        if inference_job is not None:
            raise RunningInferenceJobExistsError()

        # Get trials to load for inference job
        best_trials = self._meta_store.get_best_trials_of_train_job(train_job.id, max_count=2)
        if len(best_trials) == 0:
            raise InvalidTrainJobError('Train job has no trials with saved models!')

        # Create inference job in DB
        inference_job = self._meta_store.create_inference_job(
            user_id=user_id,
            train_job_id=train_job.id,
            budget=budget
        )
        self._meta_store.commit()

        (inference_job, predictor_service) = \
            self._services_manager.create_inference_services(inference_job.id)

        return {
            'id': inference_job.id,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'predictor_host': predictor_service.host
        }

    def stop_inference_job(self, user_id, app, app_version=-1):
        train_job = self._meta_store.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._meta_store.get_deployed_inference_job_by_train_job(train_job.id)
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
        train_job = self._meta_store.get_train_job_by_app_version(user_id, app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobError()

        inference_job = self._meta_store.get_deployed_inference_job_by_train_job(train_job.id)
        if inference_job is None:
            raise InvalidRunningInferenceJobError()
        
        predictor_service = self._meta_store.get_service(inference_job.predictor_service_id) \
                            if inference_job.predictor_service_id is not None else None

        return {
            'id': inference_job.id,
            'status': inference_job.status,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'datetime_started': inference_job.datetime_started,
            'datetime_stopped': inference_job.datetime_stopped,
            'predictor_host': predictor_service.host if predictor_service is not None else None
        }

    def get_inference_jobs_of_app(self, user_id, app):
        inference_jobs = self._meta_store.get_inference_jobs_of_app(user_id, app)
        train_jobs = [self._meta_store.get_train_job(x.train_job_id) for x in inference_jobs]
        return [
            {
                'id': inference_job.id,
                'status': inference_job.status,
                'train_job_id': train_job.id,
                'app': train_job.app,
                'app_version': train_job.app_version,
                'datetime_started': inference_job.datetime_started,
                'datetime_stopped': inference_job.datetime_stopped
            }
            for (inference_job, train_job) in zip(inference_jobs, train_jobs)
        ]

    def get_inference_jobs_by_user(self, user_id):
        inference_jobs = self._meta_store.get_inference_jobs_by_user(user_id)
        train_jobs = [self._meta_store.get_train_job(x.train_job_id) for x in inference_jobs]
        return [
            {
                'id': inference_job.id,
                'status': inference_job.status,
                'train_job_id': train_job.id,
                'app': train_job.app,
                'app_version': train_job.app_version,
                'datetime_started': inference_job.datetime_started,
                'datetime_stopped': inference_job.datetime_stopped
            }
            for (inference_job, train_job) in zip(inference_jobs, train_jobs)
        ]

    def stop_all_inference_jobs(self):
        inference_jobs = self._meta_store.get_inference_jobs_by_statuses(
            [InferenceJobStatus.STARTED, InferenceJobStatus.RUNNING]
        )
        for inference_job in inference_jobs:
            self._services_manager.stop_inference_services(inference_job.id)
            
        return [
            {
                'id': inference_job.id
            }
            for inference_job in inference_jobs
        ]


    ####################################
    # Events
    ####################################

    def handle_event(self, name, **params):
        # Call upon corresponding method of name
        try:
            method_name = f'_on_{name}'
            method = getattr(self, method_name)
            method(**params)
        except AttributeError:
            logger.error('Unknown event: "{}"'.format(name))

    def _on_sub_train_job_advisor_started(self, sub_train_job_id):
        self._services_manager.refresh_sub_train_job_status(sub_train_job_id)

    def _on_sub_train_job_advisor_stopped(self, sub_train_job_id):
        self._services_manager.refresh_sub_train_job_status(sub_train_job_id)

    def _on_sub_train_job_budget_reached(self, sub_train_job_id):
        self._services_manager.stop_sub_train_job_services(sub_train_job_id)

    def _on_train_job_worker_started(self, sub_train_job_id):
        self._services_manager.refresh_sub_train_job_status(sub_train_job_id)

    def _on_train_job_worker_stopped(self, sub_train_job_id):
        self._services_manager.refresh_sub_train_job_status(sub_train_job_id)

    def _on_inference_job_worker_started(self, inference_job_id):
        self._services_manager.refresh_inference_job_status(inference_job_id)

    def _on_inference_job_worker_stopped(self, inference_job_id):
        self._services_manager.refresh_inference_job_status(inference_job_id)

    def _on_predictor_started(self, inference_job_id):
        self._services_manager.refresh_inference_job_status(inference_job_id)

    def _on_predictor_stopped(self, inference_job_id):
        self._services_manager.refresh_inference_job_status(inference_job_id)

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, model_file_bytes, 
                    model_class, docker_image=None, dependencies={}, 
                    access_right=ModelAccessRight.PRIVATE):

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
        self._meta_store.commit()

        return {
            'id': model.id,
            'user_id': model.user_id,
            'name': model.name 
        }

    def delete_model(self, model_id):
        model = self._meta_store.get_model(model_id)
        if model is None:
            raise InvalidModelError()

        self._meta_store.delete_model(model)
        
        return {
            'id': model.id,
            'user_id': model.user_id,
            'name': model.name 
        }

    def get_model_by_name(self, user_id, name):
        model = self._meta_store.get_model_by_name(user_id, name)
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
        model = self._meta_store.get_model(model_id)
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
        model = self._meta_store.get_model(model_id)
        if model is None:
            raise InvalidModelError()

        return model.model_file_bytes

    def get_available_models(self, user_id, task=None):
        models = self._meta_store.get_available_models(user_id, task)
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
    # Private / Users
    ####################################

    def _seed_superadmin(self):
        # Seed superadmin
        try:
            self._create_user(
                email=SUPERADMIN_EMAIL,
                password=SUPERADMIN_PASSWORD,
                user_type=UserType.SUPERADMIN
            )
            logger.info('Seeded superadmin...')
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
        
