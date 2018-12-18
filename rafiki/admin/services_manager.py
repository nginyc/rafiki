import os
import logging
import traceback
import time

from rafiki.db import Database
from rafiki.constants import ServiceStatus, UserType, ServiceType, BudgetType
from rafiki.config import MIN_SERVICE_PORT, MAX_SERVICE_PORT, \
    TRAIN_WORKER_REPLICAS_PER_SUB_TRAIN_JOB, INFERENCE_WORKER_REPLICAS_PER_TRIAL, \
    INFERENCE_MAX_BEST_TRIALS, SERVICE_STATUS_WAIT
from rafiki.container import DockerSwarmContainerManager, ServiceRequirement, InvalidServiceRequest
from rafiki.model import parse_model_install_command

logger = logging.getLogger(__name__)

class ServiceDeploymentException(Exception): pass

class ServicesManager(object):
    def __init__(self, db=None, container_manager=None):
        if db is None: 
            db = Database()
        if container_manager is None: 
            container_manager = DockerSwarmContainerManager()
        
        self._postgres_host = os.environ['POSTGRES_HOST']
        self._postgres_port = os.environ['POSTGRES_PORT']
        self._postgres_user = os.environ['POSTGRES_USER']
        self._postgres_password = os.environ['POSTGRES_PASSWORD']
        self._postgres_db = os.environ['POSTGRES_DB']
        self._redis_host = os.environ['REDIS_HOST']
        self._redis_port = os.environ['REDIS_PORT']
        self._admin_host = os.environ['ADMIN_HOST']
        self._admin_host = os.environ['ADMIN_HOST']
        self._admin_port = os.environ['ADMIN_PORT']
        self._advisor_host = os.environ['ADVISOR_HOST']
        self._advisor_port = os.environ['ADVISOR_PORT']
        self._data_workdir = os.environ['DATA_WORKDIR_PATH']
        self._logs_workdir = os.environ['LOGS_WORKDIR_PATH']
        self._data_docker_workdir = os.environ['DATA_DOCKER_WORKDIR_PATH']
        self._logs_docker_workdir = os.environ['LOGS_DOCKER_WORKDIR_PATH']
        self._predictor_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_PREDICTOR'],
                                                os.environ['RAFIKI_VERSION'])
        self._predictor_port = os.environ['PREDICTOR_PORT']
        self._rafiki_addr = os.environ['RAFIKI_ADDR']

        self._db = db
        self._container_manager = container_manager

    def create_inference_services(self, inference_job_id):
        inference_job = self._db.get_inference_job(inference_job_id)

        # Prepare all inputs for inference job deployment
        # Throws error early to make service deployment more atomic
        best_trials = self._get_best_trials_for_inference(inference_job)
        trial_to_replicas = self._compute_inference_worker_replicas_for_trials(best_trials)

        try:
            # Create predictor
            predictor_service = self._create_predictor_service(inference_job)
            self._db.update_inference_job(inference_job, predictor_service_id=predictor_service.id)
            self._db.commit()

            # Create a worker service for each best trial of associated train job
            worker_services = []
            for (trial, replicas) in trial_to_replicas.items():
                service = self._create_inference_job_worker(inference_job, trial, replicas)
                worker_services.append(service)
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
        
        # Create a worker service for each model
        sub_train_job_to_replicas = self._compute_train_worker_replicas_for_sub_train_jobs(sub_train_jobs)
        worker_services = []
        for (sub_train_job, replicas) in sub_train_job_to_replicas.items():
            service = self._create_train_job_worker(train_job, sub_train_job, replicas)
            worker_services.append(service)

        # Ensure that all services are running
        self._wait_until_services_running(worker_services)

        # Mark sub train job as running
        for sub_train_job in sub_train_jobs:
            self._db.mark_sub_train_job_as_running(sub_train_job)
            
        self._db.commit()

        return train_job

    def stop_train_services(self, train_job_id):
        train_job = self._db.get_train_job(train_job_id)

        # Stop all workers for train job
        workers = self._db.get_workers_of_train_job(train_job_id)
        for worker in workers:
            self._stop_train_job_worker(worker)

        return train_job
        
    def stop_train_job_worker(self, service_id):
        train_job_service = self._db.get_train_job_worker(service_id)
        self._stop_train_job_worker(train_job_service)
        return train_job_service

    ####################################
    # Private
    ####################################

    def _create_inference_job_worker(self, inference_job, trial, replicas):
        sub_train_job = self._db.get_sub_train_job(trial.sub_train_job_id)
        model = self._db.get_model(sub_train_job.model_id)
        service_type = ServiceType.INFERENCE
        install_command = parse_model_install_command(model.dependencies, enable_gpu=False)
        environment_vars = {
            'POSTGRES_HOST': self._postgres_host,
            'POSTGRES_PORT': self._postgres_port,
            'POSTGRES_USER': self._postgres_user,
            'POSTGRES_DB': self._postgres_db,
            'POSTGRES_PASSWORD': self._postgres_password,
            'REDIS_HOST': self._redis_host,
            'REDIS_PORT': self._redis_port,
            'WORKER_INSTALL_COMMAND': install_command,
            'CUDA_VISIBLE_DEVICES': '-1' # Hide GPU
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            replicas=replicas,
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
        environment_vars = {
            'POSTGRES_HOST': self._postgres_host,
            'POSTGRES_PORT': self._postgres_port,
            'POSTGRES_USER': self._postgres_user,
            'POSTGRES_DB': self._postgres_db,
            'POSTGRES_PASSWORD': self._postgres_password,
            'REDIS_HOST': self._redis_host,
            'REDIS_PORT': self._redis_port
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=self._predictor_image,
            replicas=1,
            environment_vars=environment_vars,
            container_port=self._predictor_port
        )

        return service

    def _create_train_job_worker(self, train_job, sub_train_job, replicas):
        model = self._db.get_model(sub_train_job.model_id)
        service_type = ServiceType.TRAIN
        enable_gpu = int(train_job.budget.get(BudgetType.ENABLE_GPU, 0)) > 0
        install_command = parse_model_install_command(model.dependencies, enable_gpu=enable_gpu)
        environment_vars = {
            'POSTGRES_HOST': self._postgres_host,
            'POSTGRES_PORT': self._postgres_port,
            'POSTGRES_USER': self._postgres_user,
            'POSTGRES_DB': self._postgres_db,
            'POSTGRES_PASSWORD': self._postgres_password,
            'ADMIN_HOST': self._admin_host,
            'ADMIN_PORT': self._admin_port,
            'ADVISOR_HOST': self._advisor_host,
            'ADVISOR_PORT': self._advisor_port,
            'WORKER_INSTALL_COMMAND': install_command,
            **({'CUDA_VISIBLE_DEVICES': -1} if not enable_gpu else {}) # Hide GPU if not enabled
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

        self._db.create_train_job_worker(
            service_id=service.id,
            train_job_id=train_job.id,
            sub_train_job_id=sub_train_job.id
        )
        self._db.commit()

        return service

    def _stop_train_job_worker(self, worker):
        service = self._db.get_service(worker.service_id)
        self._stop_service(service)
        sub_train_job = self._db.get_sub_train_job(worker.sub_train_job_id)
        self._update_sub_train_job_status(sub_train_job)

    def _update_sub_train_job_status(self, sub_train_job):
        workers = self._db.get_workers_of_sub_train_job(sub_train_job.id)
        services = [self._db.get_service(x.service_id) for x in workers]
        
        # If all workers for the sub train job have stopped, stop sub train job as well
        if next((
            x for x in services 
            if x.status in [ServiceStatus.RUNNING, ServiceStatus.STARTED, ServiceStatus.DEPLOYING]
        ), None) is None:
            self._db.mark_sub_train_job_as_stopped(sub_train_job)
            self._db.commit()

    def _stop_service(self, service):
        if service.container_service_id is not None:
            self._container_manager.destroy_service(service.container_service_id)

        self._db.mark_service_as_stopped(service)
        self._db.commit()

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
                raise ServiceDeploymentException('Service of ID {} is of status {}'.format(service.id, service.status))

    def _create_service(self, service_type, docker_image,
                        replicas, environment_vars={}, args=[], 
                        container_port=None, requirements=[]):
        
        # Create service in DB
        container_manager_type = type(self._container_manager).__name__
        service = self._db.create_service(
            container_manager_type=container_manager_type,
            service_type=service_type,
            docker_image=docker_image,
            requirements=requirements
        )
        self._db.commit()

        # Pass service details as environment variables 
        environment_vars = {
            **environment_vars,
            'LOGS_DOCKER_WORKDIR_PATH': self._logs_docker_workdir,
            'RAFIKI_SERVICE_ID': service.id,
            'RAFIKI_SERVICE_TYPE': service_type
        }

        # Mount data and logs folders to containers' work directories
        mounts = {
            self._data_workdir: self._data_docker_workdir,
            self._logs_workdir: self._logs_docker_workdir
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
            
            container_service_id = container_service['id']
            hostname = container_service['hostname']
            port = container_service.get('port', None)

            self._db.mark_service_as_deploying(
                service,
                container_service_name=container_service_name,
                container_service_id=container_service_id,
                replicas=replicas,
                hostname=hostname,
                port=port,
                ext_hostname=ext_hostname,
                ext_port=ext_port
            )
            self._db.commit()

        except Exception as e:
            logger.error('Error while creating service with ID {}'.format(service.id))
            logger.error(traceback.format_exc())
            self._db.mark_service_as_errored(service)
            self._db.commit()
            raise e

        return service

    # Compute next available external port
    def _get_available_ext_port(self):
        services = self._db.get_services(status=ServiceStatus.RUNNING)
        used_ports = [int(x.ext_port) for x in services if x.ext_port is not None]
        port = MIN_SERVICE_PORT
        while port <= MAX_SERVICE_PORT:
            if port not in used_ports:
                return port

            port += 1

        return port

    def _get_best_trials_for_inference(self, inference_job):
        best_trials = self._db.get_best_trials_of_train_job(inference_job.train_job_id)
        return best_trials

    def _compute_train_worker_replicas_for_sub_train_jobs(self, sub_train_jobs):
        # TODO: Improve provisioning algorithm
        return {
            sub_train_job : TRAIN_WORKER_REPLICAS_PER_SUB_TRAIN_JOB
            for sub_train_job in sub_train_jobs
        }

    def _compute_inference_worker_replicas_for_trials(self, trials):
        # TODO: Improve provisioning algorithm
        return {
            trial : INFERENCE_WORKER_REPLICAS_PER_TRIAL
            for trial in trials
        }

    