import os
import logging
import traceback
import time
import socket
from contextlib import closing

from rafiki.db import Database
from rafiki.constants import ServiceStatus, UserType, ServiceType, BudgetType
from rafiki.config import INFERENCE_WORKER_REPLICAS_PER_TRIAL, \
    INFERENCE_MAX_BEST_TRIALS, SERVICE_STATUS_WAIT
from rafiki.container import DockerSwarmContainerManager, ContainerManager, ContainerService
from rafiki.model import parse_model_install_command

logger = logging.getLogger(__name__)

class ServiceDeploymentError(Exception): pass

# List of environment variables that will be auto-forwarded to services deployed
ENVIRONMENT_VARIABLES_AUTOFORWARD = [
    'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
    'SUPERADMIN_PASSWORD', 'POSTGRES_DB', 'REDIS_HOST', 'REDIS_PORT',
    'ADMIN_HOST', 'ADMIN_PORT', 'ADVISOR_HOST', 'ADVISOR_PORT',
    'DATA_DIR_PATH', 'LOGS_DIR_PATH', 'PARAMS_DIR_PATH', 
]
DEFAULT_TRAIN_GPU_COUNT = 0

class ServicesManager(object):
    def __init__(self, db=None, container_manager=None, 
                var_autoforward=ENVIRONMENT_VARIABLES_AUTOFORWARD):
        db = db or Database()
        container_manager: ContainerManager = container_manager or DockerSwarmContainerManager()

         # Ensure that environment variable exists, failing fast
        for x in var_autoforward:
            os.environ[x]
        self._var_autoforward = var_autoforward

        self._data_dir_path = os.environ['DATA_DIR_PATH']
        self._logs_dir_path = os.environ['LOGS_DIR_PATH']
        self._params_dir_path = os.environ['PARAMS_DIR_PATH']
        self._host_workdir_path = os.environ['HOST_WORKDIR_PATH']
        self._docker_workdir_path = os.environ['DOCKER_WORKDIR_PATH']
        self._predictor_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_PREDICTOR'],
                                                os.environ['RAFIKI_VERSION'])
        self._predictor_port = os.environ['PREDICTOR_PORT']
        self._rafiki_addr = os.environ['RAFIKI_ADDR']
        self._app_mode = os.environ['APP_MODE']

        self._db = db
        self._container_manager = container_manager

    def create_inference_services(self, inference_job_id):
        inference_job = self._db.get_inference_job(inference_job_id)

        # Prepare all inputs for inference job deployment
        # Throws error early to make service deployment more atomic
        best_trials = self._get_best_trials_for_inference(inference_job)
        trial_to_replicas = self._compute_inference_worker_replicas_for_trials(best_trials)

        try:
            # Create a worker service for each best trial of associated train job
            worker_services = []
            for (trial, replicas) in trial_to_replicas.items():
                service = self._create_inference_job_worker(inference_job, trial)
                worker_services.append(service)
                self._db.commit()

            # Create predictor
            predictor_service = self._create_predictor_service(inference_job)
            self._db.update_inference_job(inference_job, predictor_service_id=predictor_service.id)
            self._db.commit()

            # Ensure that all services are running
            self._wait_until_services_running([predictor_service, *worker_services])

            # Mark inference job as running
            self._db.mark_inference_job_as_running(inference_job)
            self._db.commit()

            return (inference_job, predictor_service)

        except Exception as e:
            # Mark inference job as errored
            self._db.mark_inference_job_as_errored(inference_job)
            self._db.commit()
            raise e
        
    def stop_inference_services(self, inference_job_id):
        inference_job = self._db.get_inference_job(inference_job_id)
        
        # Stop predictor
        service = self._db.get_service(inference_job.predictor_service_id)
        self._stop_service(service)

        # Stop all workers for inference job
        workers = self._db.get_workers_of_inference_job(inference_job_id)
        for worker in workers:
            service = self._db.get_service(worker.service_id)
            self._stop_service(service)

        self._db.mark_inference_job_as_stopped(inference_job)
        self._db.commit()

        return inference_job

    def create_train_services(self, train_job_id):
        train_job = self._db.get_train_job(train_job_id)
        sub_train_jobs = self._db.get_sub_train_jobs_of_train_job(train_job_id)

        total_gpus = int(train_job.budget.get(BudgetType.GPU_COUNT, DEFAULT_TRAIN_GPU_COUNT))
        (jobs_gpus, jobs_cpus) = self._get_deployment_for_sub_train_jobs(total_gpus, sub_train_jobs)

        # Try to create workers for each sub train job
        try:
            services = []
            for (sub_train_job, gpus, cpus) in zip(sub_train_jobs, jobs_gpus, jobs_cpus):
                # 1 GPU per worker
                for _ in range(gpus):
                    service = self._create_train_job_worker(sub_train_job)
                    services.append(service)

                # CPU workers
                for _ in range(cpus):
                    service = self._create_train_job_worker(sub_train_job, gpus=0)
                    services.append(service)
        
            self._wait_until_services_running(services)
            return train_job

        except Exception as e:
            # Stop partially started train services
            self.stop_train_services(train_job_id)
            self._db.mark_train_job_as_errored(train_job)
            raise ServiceDeploymentError(e)

    def stop_train_services(self, train_job_id):
        train_job = self._db.get_train_job(train_job_id)
        sub_train_jobs = self._db.get_sub_train_jobs_of_train_job(train_job_id)

        # Stop all sub train jobs for train job
        for sub_train_job in sub_train_jobs:
            self.stop_sub_train_job_services(sub_train_job.id)

        self._db.mark_train_job_as_stopped(train_job)

    def stop_sub_train_job_services(self, sub_train_job_id):
        sub_train_job = self._db.get_sub_train_job(sub_train_job_id)
        workers = self._db.get_workers_of_sub_train_job(sub_train_job_id)

        # Stop all workers for sub train job
        for worker in workers:
            service = self._db.get_service(worker.service_id)
            self._stop_service(service)

        self.refresh_train_job_status(sub_train_job.train_job_id)

        return sub_train_job
    
    def refresh_train_job_status(self, train_job_id):
        train_job = self._db.get_train_job(train_job_id)
        workers = self._db.get_workers_of_train_job(train_job_id)
        services = [self._db.get_service(x.service_id) for x in workers]

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
            self._db.mark_train_job_as_errored(train_job)
        elif count[ServiceStatus.STOPPED] == len(services):
            self._db.mark_train_job_as_stopped(train_job)
        elif count[ServiceStatus.RUNNING] > 0:
            self._db.mark_train_job_as_running(train_job)

    ####################################
    # Private
    ####################################

    def _get_deployment_for_sub_train_jobs(self, total_gpus, sub_train_jobs):
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

    def _create_inference_job_worker(self, inference_job, trial):
        sub_train_job = self._db.get_sub_train_job(trial.sub_train_job_id)
        model = self._db.get_model(sub_train_job.model_id)
        service_type = ServiceType.INFERENCE
        install_command = parse_model_install_command(model.dependencies, enable_gpu=False)
        environment_vars = {
            'WORKER_INSTALL_COMMAND': install_command,
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            environment_vars=environment_vars
        )

        self._db.create_inference_job_worker(
            service_id=service.id,
            inference_job_id=inference_job.id,
            trial_id=trial.id
        )
        self._db.commit()

        return service

    def _create_predictor_service(self, inference_job):
        service_type = ServiceType.PREDICT
        environment_vars = {}

        service = self._create_service(
            service_type=service_type,
            docker_image=self._predictor_image,
            environment_vars=environment_vars,
            container_port=self._predictor_port
        )

        return service

    def _create_train_job_worker(self, sub_train_job, gpus=1):
        model = self._db.get_model(sub_train_job.model_id)
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

        self._db.create_train_job_worker(
            service_id=service.id,
            sub_train_job_id=sub_train_job.id
        )
        self._db.commit()

        return service

    def _stop_service(self, service):
        if service.status == ServiceStatus.STOPPED:
            logger.info('Service of ID "{}" already stopped!'.format(service.id))
            return

        try:
            container_service = self._get_container_service_from_service(service)
            self._container_manager.destroy_service(container_service)
            self._db.mark_service_as_stopped(service)
            self._db.commit()
        except Exception:
            # Allow exception to be thrown if deleting service fails (maybe concurrent service deletion)
            logger.info('Error while deleting service with ID {} - maybe already deleted'.format(service.id))
            logger.info(traceback.format_exc())

    # Returns when all services have status of `RUNNING`
    # Throws an exception if any of the services have a status of `ERRORED` or `STOPPED`
    def _wait_until_services_running(self, services):
        for service in services:
            while service.status not in \
                    [ServiceStatus.RUNNING, ServiceStatus.ERRORED, ServiceStatus.STOPPED]:
                time.sleep(SERVICE_STATUS_WAIT)
                self._db.expire()
                service = self._db.get_service(service.id)

            if service.status in [ServiceStatus.ERRORED, ServiceStatus.STOPPED]:
                raise ServiceDeploymentError('Service of ID {} is of status {}'.format(service.id, service.status))

    def _create_service(self, service_type, docker_image,
                        replicas=1, environment_vars={}, args=[], 
                        container_port=None, gpus=0):
        
        # Create service in DB
        container_manager_type = type(self._container_manager).__name__
        service = self._db.create_service(
            container_manager_type=container_manager_type,
            service_type=service_type,
            docker_image=docker_image,
            replicas=replicas,
            gpus=gpus
        )
        self._db.commit()

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
            container_service_name = 'rafiki_service_{}'.format(service.id)
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
            
            self._db.mark_service_as_deploying(
                service,
                container_service_name=container_service_name,
                container_service_id=container_service.id,
                hostname=container_service.hostname,
                port=container_service.port,
                ext_hostname=ext_hostname,
                ext_port=ext_port,
                container_service_info=container_service.info
            )
            self._db.commit()

        except Exception as e:
            logger.error('Error while creating service with ID {}'.format(service.id))
            logger.error(traceback.format_exc())
            self._db.mark_service_as_errored(service)
            self._db.commit()
            raise e

        return service

    def _get_available_ext_port(self):
        # Credits to https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    def _get_best_trials_for_inference(self, inference_job):
        best_trials = self._db.get_best_trials_of_train_job(inference_job.train_job_id)
        return best_trials

    def _compute_inference_worker_replicas_for_trials(self, trials):
        # TODO: Improve provisioning algorithm
        return {
            trial : INFERENCE_WORKER_REPLICAS_PER_TRIAL
            for trial in trials
        }

    def _get_container_service_from_service(self, service):
        service_id = service.container_service_id
        hostname = service.hostname
        port = service.port
        info = service.container_service_info
        container_service = ContainerService(service_id, hostname, port, info)
        return container_service