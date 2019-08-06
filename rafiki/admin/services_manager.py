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
import socket
from collections import defaultdict
from contextlib import closing

from rafiki.constants import ServiceStatus, ServiceType, BudgetOption, InferenceBudgetOption, TrainJobStatus, InferenceJobStatus
from rafiki.meta_store import MetaStore
from rafiki.container import DockerSwarmContainerManager, ContainerManager, ContainerService
from rafiki.model import parse_model_install_command

logger = logging.getLogger(__name__)

class ServiceDeploymentError(Exception): pass

# List of environment variables that will be auto-forwarded to services deployed
ENVIRONMENT_VARIABLES_AUTOFORWARD = [
    'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
    'SUPERADMIN_PASSWORD', 'POSTGRES_DB', 'REDIS_HOST', 'REDIS_PORT',
    'ADMIN_HOST', 'ADMIN_PORT', 'DATA_DIR_PATH', 'LOGS_DIR_PATH', 'PARAMS_DIR_PATH', 'KAFKA_HOST', 'KAFKA_PORT',
]
DEFAULT_TRAIN_GPU_COUNT = 0
DEFAULT_INFERENCE_GPU_COUNT = 0
SERVICE_STATUS_WAIT_SECS = 1

class ServicesManager(object):
    '''
        Manages deployment of services and statuses of train jobs & inference jobs
    '''

    def __init__(self, meta_store=None, container_manager=None, 
                var_autoforward=ENVIRONMENT_VARIABLES_AUTOFORWARD):
        self._meta_store: MetaStore = meta_store or MetaStore()
        self._container_manager: ContainerManager = container_manager or DockerSwarmContainerManager()

         # Ensure that environment variable exists, failing fast
        for x in var_autoforward:
            os.environ[x]
        self._var_autoforward = var_autoforward

        version = os.environ['RAFIKI_VERSION']
        self._data_dir_path = os.environ['DATA_DIR_PATH']
        self._logs_dir_path = os.environ['LOGS_DIR_PATH']
        self._params_dir_path = os.environ['PARAMS_DIR_PATH']
        self._host_workdir_path = os.environ['HOST_WORKDIR_PATH']
        self._docker_workdir_path = os.environ['DOCKER_WORKDIR_PATH']
        self._predictor_image = f"{os.environ['RAFIKI_IMAGE_PREDICTOR']}:{version}"
        self._predictor_port = os.environ['PREDICTOR_PORT']
        self._app_mode = os.environ['APP_MODE']
        self._rafiki_addr = os.environ['RAFIKI_ADDR']
        self._app_mode = os.environ['APP_MODE']

    def create_inference_services(self, inference_job_id):
        inference_job = self._meta_store.get_inference_job(inference_job_id)
        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(inference_job.train_job_id)

        # Determine trials to be deployed & GPU allocation for these trials
        total_gpus = int(inference_job.budget.get(InferenceBudgetOption.GPU_COUNT, DEFAULT_INFERENCE_GPU_COUNT))
        (trial_ids, jobs_gpus) = self._get_deployment_for_inference_job(total_gpus, sub_train_jobs)

        try:
            # Create predictor
            predictor_service = self._create_predictor(inference_job)

            # Create worker for each trial to be deployed
            for (trial_id, gpus) in zip(trial_ids, jobs_gpus):
                trial = self._meta_store.get_trial(trial_id)
                self._create_inference_job_worker(inference_job, trial, gpus=gpus)

            return (inference_job, predictor_service)

        except Exception as e:
            # Stop partially started inference services
            self.stop_inference_services(inference_job_id)
            self._meta_store.mark_inference_job_as_errored(inference_job)
            raise ServiceDeploymentError(e)   

    def stop_inference_services(self, inference_job_id):
        inference_job = self._meta_store.get_inference_job(inference_job_id)
        
        # Stop predictor
        if inference_job.predictor_service_id is not None:
            service = self._meta_store.get_service(inference_job.predictor_service_id)
            self._stop_service(service)

        # Stop all workers for inference job
        workers = self._meta_store.get_workers_of_inference_job(inference_job_id)
        for worker in workers:
            service = self._meta_store.get_service(worker.service_id)
            self._stop_service(service)

        self.refresh_inference_job_status(inference_job_id)

        return inference_job

    def refresh_inference_job_status(self, inference_job_id):
        inference_job = self._meta_store.get_inference_job(inference_job_id)
        assert inference_job is not None

        # If inference job once errored, keep it errored
        if inference_job.status == InferenceJobStatus.ERRORED:
            return

        predictor_service_id = inference_job.predictor_service_id
        workers = self._meta_store.get_workers_of_inference_job(inference_job_id)
        predictor_service = self._meta_store.get_service(predictor_service_id) if predictor_service_id is not None else None
        services = [self._meta_store.get_service(x.service_id) for x in workers]

        # Count statuses of workers
        worker_counts = defaultdict(int)
        for service in services:
            if service is not None:
                worker_counts[service.status] += 1
        predictor_status = predictor_service.status if predictor_service is not None else None

        # If predictor is running and at least 1 worker, it is running
        # If predictor is stopped, it is stopped
        # If predictor is errored or all workers are errored, it is errored, and stop all services
        if predictor_status == ServiceStatus.RUNNING and worker_counts[ServiceStatus.RUNNING] >= 1:
            self._meta_store.mark_inference_job_as_running(inference_job)
        elif predictor_status == ServiceStatus.STOPPED:
            self._meta_store.mark_inference_job_as_stopped(inference_job)
        elif predictor_status == ServiceStatus.ERRORED or \
                worker_counts[ServiceStatus.ERRORED] == len(workers):
            self._meta_store.mark_inference_job_as_errored(inference_job)
            self.stop_inference_services(inference_job_id)

        self._meta_store.commit()

    def create_train_services(self, train_job_id):
        train_job = self._meta_store.get_train_job(train_job_id)
        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(train_job_id)

        # Determine CPU & GPU allocation across sub train jobs
        total_gpus = int(train_job.budget.get(BudgetOption.GPU_COUNT, DEFAULT_TRAIN_GPU_COUNT))
        (jobs_gpus, jobs_cpus) = self._get_deployment_for_train_job(total_gpus, sub_train_jobs)

        # Try to create advisors & workers for each sub train job
        try:
            for (sub_train_job, gpus, cpus) in zip(sub_train_jobs, jobs_gpus, jobs_cpus):
                # Create advisor
                self._create_advisor(sub_train_job)
                
                # 1 GPU per worker
                for _ in range(gpus):
                    self._create_train_job_worker(sub_train_job)

                # CPU workers
                for _ in range(cpus):
                    self._create_train_job_worker(sub_train_job, gpus=0)
        
            return train_job

        except Exception as e:
            # Stop partially started train services
            self.stop_train_services(train_job_id)
            self._meta_store.mark_train_job_as_errored(train_job)
            raise ServiceDeploymentError(e)

    def stop_train_services(self, train_job_id):
        train_job = self._meta_store.get_train_job(train_job_id)
        assert train_job is not None
        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(train_job_id)

        # Stop all sub train jobs for train job
        for sub_train_job in sub_train_jobs:
            self.stop_sub_train_job_services(sub_train_job.id)

    def stop_sub_train_job_services(self, sub_train_job_id):
        sub_train_job = self._meta_store.get_sub_train_job(sub_train_job_id)
        assert sub_train_job is not None
        workers = self._meta_store.get_workers_of_sub_train_job(sub_train_job_id)

        # Stop advisor for sub train job
        if sub_train_job.advisor_service_id is not None:
            service = self._meta_store.get_service(sub_train_job.advisor_service_id)
            self._stop_service(service)

        # Stop all workers for sub train job
        for worker in workers:
            service = self._meta_store.get_service(worker.service_id)
            self._stop_service(service)

        self.refresh_sub_train_job_status(sub_train_job_id)

        return sub_train_job

    def refresh_sub_train_job_status(self, sub_train_job_id):
        sub_train_job = self._meta_store.get_sub_train_job(sub_train_job_id)
        assert sub_train_job is not None

        # If sub train job once errored, keep it errored
        if sub_train_job.status == TrainJobStatus.ERRORED:
            return

        advisor_service_id = sub_train_job.advisor_service_id
        workers = self._meta_store.get_workers_of_sub_train_job(sub_train_job_id)
        advisor_service = self._meta_store.get_service(advisor_service_id) if advisor_service_id is not None else None
        services = [self._meta_store.get_service(x.service_id) for x in workers]

        # Count statuses of workers
        worker_counts = defaultdict(int)
        for service in services:
            if service is not None:
                worker_counts[service.status] += 1
        advisor_status = advisor_service.status if advisor_service is not None else None

        # If advisor is running and at least 1 worker, it is running
        # If advisor is stopped, it is stopped
        # If advisor is errored or all workers are errored, it is errored, and stop all services
        if advisor_status == ServiceStatus.RUNNING and worker_counts[ServiceStatus.RUNNING] >= 1:
            self._meta_store.mark_sub_train_job_as_running(sub_train_job)
        elif advisor_status == ServiceStatus.STOPPED:
            self._meta_store.mark_sub_train_job_as_stopped(sub_train_job)
        elif advisor_status == ServiceStatus.ERRORED or \
                worker_counts[ServiceStatus.ERRORED] == len(workers):
            self._meta_store.mark_sub_train_job_as_errored(sub_train_job)
            self.stop_sub_train_job_services(sub_train_job_id)

        self._meta_store.commit()

        # Refresh train job status (which depends sub train job statuses)
        self.refresh_train_job_status(sub_train_job.train_job_id)

    def refresh_train_job_status(self, train_job_id):
        train_job = self._meta_store.get_train_job(train_job_id)
        assert train_job is not None
        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(train_job_id)

        # If train job once errored, keep it errored
        if train_job.status == TrainJobStatus.ERRORED:
            return

        # Count statuses of sub train jobs
        counts = defaultdict(int)
        for job in sub_train_jobs:
            counts[job.status] += 1

        # If any sub train job is running, train job is running
        # If all sub train jobs are stopped, train job is stopped
        # If any sub train job is errored and all others have stopped, train job is errored
        if counts[TrainJobStatus.RUNNING] >= 1:
            self._meta_store.mark_train_job_as_running(train_job)
        elif counts[TrainJobStatus.STOPPED] == len(sub_train_jobs):
            self._meta_store.mark_train_job_as_stopped(train_job)
        elif counts[TrainJobStatus.ERRORED] >= 1 and \
            (counts[TrainJobStatus.ERRORED] + counts[TrainJobStatus.STOPPED]) == len(sub_train_jobs):
            self._meta_store.mark_train_job_as_errored(train_job)

    ####################################
    # Private
    ####################################

    def _get_deployment_for_train_job(self, total_gpus, sub_train_jobs):
        # Evenly distribute GPus across sub train jobs, letting first few sub train jobs have 1 more GPU to fully allocate
        N = len(sub_train_jobs)
        base_gpus = total_gpus // N
        extra_gpus = total_gpus - base_gpus * N
        jobs_gpus = ([base_gpus + 1] * extra_gpus) + [base_gpus] * (N - extra_gpus)

        # For jobs with no GPU, add 1 CPU
        jobs_cpus = []
        for gpus in jobs_gpus:
            jobs_cpus.append(0 if gpus > 0 else 1)

        return (jobs_gpus, jobs_cpus)

    def _get_deployment_for_inference_job(self, total_gpus, sub_train_jobs):
        trial_ids = []
        # For each sub train job, to deploy best-scoring trial
        for sub_train_job in sub_train_jobs:
            trials = self._meta_store.get_best_trials_of_sub_train_job(sub_train_job.id, max_count=1)
            if len(trials) == 0:
                continue
            trial_ids.append(trials[0].id)
        
        # Evenly distribute GPus across trials, letting first few trials have 1 more GPU to fully allocate
        N = len(trial_ids)
        base_gpus = total_gpus // N
        extra_gpus = total_gpus - base_gpus * N
        jobs_gpus = ([base_gpus + 1] * extra_gpus) + [base_gpus] * (N - extra_gpus)

        return (trial_ids, jobs_gpus)

    def _create_inference_job_worker(self, inference_job, trial, gpus=0):
        sub_train_job = self._meta_store.get_sub_train_job(trial.sub_train_job_id)
        model = self._meta_store.get_model(sub_train_job.model_id)
        service_type = ServiceType.INFERENCE
        install_command = parse_model_install_command(model.dependencies, enable_gpu=(gpus > 0))
        environment_vars = {
            'WORKER_INSTALL_COMMAND': install_command,
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            environment_vars=environment_vars,
            gpus=gpus
        )

        self._meta_store.create_inference_job_worker(
            service_id=service.id,
            inference_job_id=inference_job.id,
            trial_id=trial.id
        )
        self._meta_store.commit()

        return service

    def _create_predictor(self, inference_job):
        service_type = ServiceType.PREDICT
        environment_vars = {}

        service = self._create_service(
            service_type=service_type,
            docker_image=self._predictor_image,
            environment_vars=environment_vars,
            container_port=self._predictor_port
        )

        self._meta_store.update_inference_job(inference_job, predictor_service_id=service.id)
        self._meta_store.commit()

        return service

    def _create_train_job_worker(self, sub_train_job, gpus=1):
        model = self._meta_store.get_model(sub_train_job.model_id)
        service_type = ServiceType.TRAIN
        install_command = parse_model_install_command(model.dependencies, enable_gpu=(gpus > 0))
        environment_vars = {
            'WORKER_INSTALL_COMMAND': install_command,
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            environment_vars=environment_vars,
            gpus=gpus
        )

        self._meta_store.create_train_job_worker(
            service_id=service.id,
            sub_train_job_id=sub_train_job.id
        )
        self._meta_store.commit()

        return service

    def _create_advisor(self, sub_train_job):
        model = self._meta_store.get_model(sub_train_job.model_id)
        service_type = ServiceType.ADVISOR
        install_command = parse_model_install_command(model.dependencies, enable_gpu=False)
        environment_vars = {
            'WORKER_INSTALL_COMMAND': install_command,
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            environment_vars=environment_vars
        )

        self._meta_store.update_sub_train_job(sub_train_job, advisor_service_id=service.id)
        self._meta_store.commit()

        return service

    def _stop_service(self, service):
        if service.status == ServiceStatus.STOPPED:
            logger.info('Service of ID "{}" already stopped!'.format(service.id))
            return

        try:
            container_service = self._get_container_service_from_service(service)
            self._container_manager.destroy_service(container_service)
        except:
            # Allow exception to be thrown if deleting service fails (maybe concurrent service deletion)
            logger.info('Error while deleting service with ID {} - maybe already deleted'.format(service.id))
            logger.info(traceback.format_exc())

        self._meta_store.mark_service_as_stopped(service)
        self._meta_store.commit()
        
    def _create_service(self, service_type, docker_image,
                        replicas=1, environment_vars={}, args=[], 
                        container_port=None, gpus=0):
        
        # Create service in DB
        container_manager_type = type(self._container_manager).__name__
        service = self._meta_store.create_service(
            service_type=service_type,
            container_manager_type=container_manager_type,
            docker_image=docker_image,
            replicas=replicas,
            gpus=gpus
        )
        self._meta_store.commit()

        # Pass service details as environment variables 
        environment_vars = {
            # Autofoward environment variables
            **{
                x: os.environ[x]
                for x in self._var_autoforward
            },
            **environment_vars,
            'RAFIKI_SERVICE_ID': service.id,
            'RAFIKI_SERVICE_TYPE': service_type,
            'WORKDIR_PATH': self._docker_workdir_path
        }

        if self._app_mode == 'DEV':
            # Mount whole root directory
            mounts = {
                self._host_workdir_path: self._docker_workdir_path
            }
        else:
            # Mount only data, logs and params folders to containers' work directories
            mounts = {
                os.path.join(self._host_workdir_path, self._data_dir_path): 
                    os.path.join(self._docker_workdir_path, self._data_dir_path),
                os.path.join(self._host_workdir_path, self._logs_dir_path): 
                    os.path.join(self._docker_workdir_path, self._logs_dir_path),
                os.path.join(self._host_workdir_path, self._params_dir_path): 
                    os.path.join(self._docker_workdir_path, self._params_dir_path)
            }

        # Expose container port if it exists
        publish_port = None
        ext_hostname = None
        ext_port = None
        if container_port is not None:
            ext_hostname = self._rafiki_addr
            ext_port = self._get_available_ext_port()
            publish_port = (ext_port, container_port)

        try:
            container_service_name = ''
            if os.getenv('CONTAINER_MODE', 'SWARM') == 'SWARM':
                container_service_name = 'rafiki_service_{}'.format(service.id)
            else:
                container_service_name = 'rafiki-service-{}'.format(service.id)
            container_service = self._container_manager.create_service(
                service_name=container_service_name,
                docker_image=docker_image, 
                replicas=replicas, 
                args=args,
                environment_vars=environment_vars,
                mounts=mounts,
                publish_port=publish_port,
                gpus=gpus
            )
            
            self._meta_store.mark_service_as_deploying(
                service,
                container_service_name=container_service_name,
                container_service_id=container_service.id,
                hostname=container_service.hostname,
                port=container_service.port,
                ext_hostname=ext_hostname,
                ext_port=ext_port,
                container_service_info=container_service.info
            )
            self._meta_store.commit()

        except Exception as e:
            logger.error('Error while creating service with ID {}'.format(service.id))
            logger.error(traceback.format_exc())
            self._meta_store.mark_service_as_errored(service)
            self._meta_store.commit()
            raise e

        return service

    def _get_available_ext_port(self):
        # Credits to https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    def _get_container_service_from_service(self, service):
        service_id = service.container_service_id
        hostname = service.hostname
        port = service.port
        info = service.container_service_info
        container_service = ContainerService(service_id, hostname, port, info)
        return container_service
