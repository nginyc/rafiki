import os
import logging
import traceback
import time
import socket
from contextlib import closing

from rafiki.meta_store import MetaStore
from rafiki.constants import ServiceStatus, UserType, ServiceType, BudgetType
from rafiki.config import TRAIN_WORKER_REPLICAS_PER_SUB_TRAIN_JOB, INFERENCE_WORKER_REPLICAS_PER_TRIAL, \
    INFERENCE_MAX_BEST_TRIALS, SERVICE_STATUS_WAIT
from rafiki.container import DockerSwarmContainerManager, ServiceRequirement, \
                            InvalidServiceRequestError, ContainerService
from rafiki.model import parse_model_install_command

logger = logging.getLogger(__name__)

class ServiceDeploymentException(Exception): pass

# List of environment variables that will be auto-forwarded to services deployed
ENVIRONMENT_VARIABLES_AUTOFORWARD = [
    'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
    'SUPERADMIN_PASSWORD', 'POSTGRES_DB', 'REDIS_HOST', 'REDIS_PORT',
    'ADMIN_HOST', 'ADMIN_PORT', 'ADVISOR_HOST', 'ADVISOR_PORT',
    'DATA_DIR_PATH', 'LOGS_DIR_PATH', 'PARAMS_DIR_PATH', 
]

class ServicesManager(object):
    def __init__(self, meta_store=None, container_manager=None,
                var_autoforward=ENVIRONMENT_VARIABLES_AUTOFORWARD):
        if meta_store is None: 
            meta_store = MetaStore()
        if container_manager is None: 
            container_manager = DockerSwarmContainerManager()

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
        self._app_mode = os.environ['APP_MODE']
        self._rafiki_addr = os.environ['RAFIKI_ADDR']

        self._meta_store = meta_store
        self._container_manager = container_manager

    def create_inference_services(self, inference_job_id):
        inference_job = self._meta_store.get_inference_job(inference_job_id)
        sub_train_jobs = self._meta_store.get_sub_inference_jobs_of_inference_job(inference_job_id)

        services_to_wait = []

        # Create a worker service for each sub inference job, wait for them to be running, and associate service to job
        job_to_replicas = self._compute_inference_worker_replicas_for_sub_inference_jobs(sub_train_jobs)
        for (sub_inference_job, replicas) in job_to_replicas.items():
            service = self._create_sub_inference_job_service(inference_job, sub_inference_job, replicas)
            self._meta_store.update_sub_inference_job(sub_inference_job, service_id=service.id)
            self._meta_store.commit()
            services_to_wait.append(service)

        # Create predictor for inference job
        predictor_service = self._create_predictor_service(inference_job)
        self._meta_store.update_inference_job(inference_job, predictor_service_id=predictor_service.id)
        self._meta_store.commit()
        services_to_wait.append(predictor_service)

        # Wait for services to return 
        self._wait_until_services_running(services_to_wait)

        return (inference_job, predictor_service)

    def stop_inference_services(self, inference_job_id):
        inference_job = self._meta_store.get_inference_job(inference_job_id)
        
        # Stop predictor
        service = self._meta_store.get_service(inference_job.predictor_service_id)
        self._stop_service(service)

        # Stop all workers for inference job
        sub_inference_jobs = self._meta_store.get_sub_inference_jobs_of_inference_job(inference_job_id)
        for sub_inference_job in sub_inference_jobs:
            service = self._meta_store.get_service(sub_inference_job.service_id)
            self._stop_service(service)

        return inference_job

    def create_train_services(self, train_job_id):
        train_job = self._meta_store.get_train_job(train_job_id)
        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(train_job_id)
        
        # Create a worker service for each sub train job, wait for them to be running, and associate service to job
        sub_train_job_to_replicas = self._compute_train_worker_replicas_for_sub_train_jobs(sub_train_jobs)
        for (sub_train_job, replicas) in sub_train_job_to_replicas.items():
            service = self._create_sub_train_job_service(train_job, sub_train_job, replicas)
            self._meta_store.update_sub_train_job(sub_train_job, service_id=service.id)
            self._meta_store.commit()
            self._wait_until_services_running([service])

        return train_job

    def stop_train_services(self, train_job_id):
        train_job = self._meta_store.get_train_job(train_job_id)

        # Stop all workers for train job
        sub_train_jobs = self._meta_store.get_sub_train_jobs_of_train_job(train_job_id)
        for sub_train_job in sub_train_jobs:
            self.stop_sub_train_job_services(sub_train_job.id)

        return train_job
        
    def stop_sub_train_job_services(self, sub_train_job_id):
        sub_train_job = self._meta_store.get_sub_train_job(sub_train_job_id)
        service = self._meta_store.get_service(sub_train_job.service_id)
        self._stop_service(service)
        return sub_train_job

    ####################################
    # Private
    ####################################

    def _create_sub_inference_job_service(self, inference_job, sub_inference_job, replicas):
        trial = self._meta_store.get_trial(sub_inference_job.trial_id)
        model = self._meta_store.get_model(trial.model_id)

        service_type = ServiceType.INFERENCE
        install_command = parse_model_install_command(model.dependencies, enable_gpu=False)
        environment_vars = {
            'WORKER_INSTALL_COMMAND': install_command
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            replicas=replicas,
            environment_vars=environment_vars
        )

        return service

    def _create_predictor_service(self, inference_job):
        service_type = ServiceType.PREDICT
        environment_vars = {}

        service = self._create_service(
            service_type=service_type,
            docker_image=self._predictor_image,
            replicas=1,
            environment_vars=environment_vars,
            container_port=self._predictor_port
        )

        return service

    def _create_sub_train_job_service(self, train_job, sub_train_job, replicas):
        model = self._meta_store.get_model(sub_train_job.model_id)

        service_type = ServiceType.TRAIN
        enable_gpu = int(train_job.budget.get(BudgetType.ENABLE_GPU, 0)) > 0
        install_command = parse_model_install_command(model.dependencies, enable_gpu=enable_gpu)
        environment_vars = {
            'WORKER_INSTALL_COMMAND': install_command,
        }

        requirements = []
        if enable_gpu:
            requirements.append(ServiceRequirement.GPU)

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            replicas=replicas,
            environment_vars=environment_vars,
            requirements=requirements
        )

        return service

    def _stop_service(self, service):
        if service.status == ServiceStatus.STOPPED:
            logger.info('Service of ID "{}" already stopped!'.format(service.id))
            return

        try:
            container_service = self._get_container_service_from_service(service)
            self._container_manager.destroy_service(container_service)
            self._meta_store.mark_service_as_stopped(service)
            self._meta_store.commit()
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
                self._meta_store.expire()
                service = self._meta_store.get_service(service.id)

            if service.status in [ServiceStatus.ERRORED, ServiceStatus.STOPPED]:
                raise ServiceDeploymentException('Service of ID {} is of status {}'.format(service.id, service.status))

    def _create_service(self, service_type, docker_image,
                        replicas, environment_vars={}, args=[], 
                        container_port=None, requirements=[]):
        
        # Create service in DB
        container_manager_type = type(self._container_manager).__name__
        service = self._meta_store.create_service(
            container_manager_type=container_manager_type,
            service_type=service_type,
            docker_image=docker_image,
            requirements=requirements
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
            container_service_name = 'rafiki_service_{}'.format(service.id)
            container_service = self._container_manager.create_service(
                service_name=container_service_name,
                docker_image=docker_image, 
                replicas=replicas, 
                args=args,
                environment_vars=environment_vars,
                mounts=mounts,
                publish_port=publish_port,
                requirements=requirements
            )
            
            self._meta_store.mark_service_as_deploying(
                service,
                container_service_name=container_service_name,
                container_service_id=container_service.id,
                replicas=replicas,
                hostname=container_service.hostname,
                port=container_service.port,
                ext_hostname=ext_hostname,
                ext_port=ext_port,
                service_info = container_service.info
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

    def _compute_train_worker_replicas_for_sub_train_jobs(self, sub_train_jobs):
        # TODO: Improve provisioning algorithm
        return {
            sub_train_job : TRAIN_WORKER_REPLICAS_PER_SUB_TRAIN_JOB
            for sub_train_job in sub_train_jobs
        }

    def _compute_inference_worker_replicas_for_sub_inference_jobs(self, sub_inference_jobs):
        # TODO: Improve provisioning algorithm
        return {
            sub_inference_job : INFERENCE_WORKER_REPLICAS_PER_TRIAL
            for sub_inference_job in sub_inference_jobs
        }

    def _get_container_service_from_service(self, service):
        service_id = service.container_service_id
        hostname = service.hostname
        port = service.port
        info = service.service_info
        container_service = ContainerService(service_id, hostname, port, info)
        return container_service
    