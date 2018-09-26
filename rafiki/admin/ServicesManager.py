import numpy as np
import os
import logging
import traceback

from rafiki.db import Database
from rafiki.constants import ServiceStatus, UserType, ServiceType
from rafiki.config import MIN_SERVICE_PORT, MAX_SERVICE_PORT

from .containers import DockerSwarmContainerManager 

logger = logging.getLogger(__name__)

class ServicesManager(object):
    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._query_frontend_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_QUERY_FRONTEND'],
                                                os.environ['RAFIKI_VERSION'])
        self._query_frontend_port = os.environ['QUERY_FRONTEND_PORT']

        self._db = db
        self._container_manager = container_manager

    def create_inference_services(self, inference_job_id):
        inference_job = self._db.get_inference_job(inference_job_id)
        
        # Create query frontend
        query_service = self._create_query_service(inference_job)

        # Create a worker service for each best trial of associated train job
        best_trials = self._db.get_best_trials_of_train_job(inference_job.train_job_id, max_count=2)
        trial_to_replicas = self._compute_inference_worker_replicas_for_trials(best_trials)
        for (trial, replicas) in trial_to_replicas.items():
            self._create_inference_job_worker(inference_job, trial, replicas)

        self._db.mark_inference_job_as_running(inference_job, query_service_id=query_service.id)
        self._db.commit()

        return (inference_job, query_service)

    def stop_inference_services(self, inference_job_id):
        inference_job = self._db.get_inference_job(inference_job_id)
        
        # Stop query frontend
        service = self._db.get_service(inference_job.query_service_id)
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
        models = self._db.get_models_of_task(train_job.task)
        model_to_replicas = self._compute_train_worker_replicas_for_models(models)

        # Create a worker service for each model
        for (model, replicas) in model_to_replicas.items():
            self._create_train_job_worker(train_job, model, replicas)

        self._update_train_job_status(train_job)
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
        model = self._db.get_model(trial.model_id)
        service_type = ServiceType.INFERENCE
        environment_vars = {
            'POSTGRES_HOST': os.environ['POSTGRES_HOST'],
            'POSTGRES_PORT': os.environ['POSTGRES_PORT'],
            'POSTGRES_USER': os.environ['POSTGRES_USER'],
            'POSTGRES_DB': os.environ['POSTGRES_DB'],
            'POSTGRES_PASSWORD': os.environ['POSTGRES_PASSWORD'],
            'LOGS_FOLDER_PATH': os.environ['LOGS_FOLDER_PATH'],
            'REDIS_HOST': os.environ['REDIS_HOST'],
            'REDIS_PORT': os.environ['REDIS_PORT']
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

    def _create_query_service(self, inference_job):
        service_type = ServiceType.QUERY
        environment_vars = {
            'POSTGRES_HOST': os.environ['POSTGRES_HOST'],
            'POSTGRES_PORT': os.environ['POSTGRES_PORT'],
            'POSTGRES_USER': os.environ['POSTGRES_USER'],
            'POSTGRES_DB': os.environ['POSTGRES_DB'],
            'POSTGRES_PASSWORD': os.environ['POSTGRES_PASSWORD'],
            'LOGS_FOLDER_PATH': os.environ['LOGS_FOLDER_PATH'],
            'REDIS_HOST': os.environ['REDIS_HOST'],
            'REDIS_PORT': os.environ['REDIS_PORT']
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=self._query_frontend_image,
            replicas=1,
            environment_vars=environment_vars,
            container_port=self._query_frontend_port
        )

        return service

    def _create_train_job_worker(self, train_job, model, replicas):
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
            'ADVISOR_HOST': os.environ['ADVISOR_HOST'],
            'ADVISOR_PORT': os.environ['ADVISOR_PORT']
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            replicas=replicas,
            environment_vars=environment_vars
        )

        self._db.create_train_job_worker(
            service_id=service.id,
            train_job_id=train_job.id,
            model_id=model.id
        )
        self._db.commit()

        return service

    def _stop_train_job_worker(self, worker):
        service = self._db.get_service(worker.service_id)
        self._stop_service(service)
        train_job = self._db.get_train_job(worker.train_job_id)
        self._update_train_job_status(train_job)

    def _update_train_job_status(self, train_job):
        workers = self._db.get_workers_of_train_job(train_job.id)
        services = [self._db.get_service(x.service_id) for x in workers]
        
        # If all workers for the train job have stopped, stop train job as well
        if next((
            x for x in services 
            if x.status in [ServiceStatus.RUNNING, ServiceStatus.STARTED]
        ), None) is None:
            self._db.mark_train_job_as_complete(train_job)
            self._db.commit()

        # If any worker for the train job is running, mark train job as running
        elif next((
            x for x in services 
            if x.status in [ServiceStatus.RUNNING]
        ), None) is not None:
            self._db.mark_train_job_as_running(train_job)
            self._db.commit()

    def _stop_service(self, service):
        if service.container_service_id is not None:
            self._container_manager.destroy_service(service.container_service_id)

        self._db.mark_service_as_stopped(service)
        self._db.commit()

    def _create_service(self, service_type, docker_image,
                        replicas, environment_vars={}, args=[], 
                        container_port=None):
        
        # Create service in DB
        container_manager_type = type(self._container_manager).__name__
        service = self._db.create_service(
            container_manager_type=container_manager_type,
            service_type=service_type,
            docker_image=docker_image
        )
        self._db.commit()

        # Pass service details as environment variables 
        environment_vars = {
            **environment_vars,
            'RAFIKI_SERVICE_ID': service.id,
            'RAFIKI_SERVICE_TYPE': service_type
        }

        # Mount logs folder onto workers too
        logs_folder_path = os.environ['LOGS_FOLDER_PATH']
        mounts = {
            logs_folder_path: logs_folder_path
        }

        # Expose container port if it exists
        publish_port = None
        ext_hostname = None
        ext_port = None
        if container_port is not None:
            ext_hostname = os.environ['RAFIKI_IP_ADDRESS']
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
                publish_port=publish_port
            )
            
            container_service_id = container_service['id']
            hostname = container_service['hostname']
            port = container_service.get('port', None)

            self._db.mark_service_as_running(
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

        except Exception:
            logger.error('Error while creating service with ID {}'.format(service.id))
            logger.error(traceback.format_exc())
            self._db.mark_service_as_errored(service)
            self._db.commit()

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

    def _compute_train_worker_replicas_for_models(self, models):
        # TODO: Improve provisioning algorithm
        return {
            model : 2 # 2 replicas per model
            for model in models
        }

    def _compute_inference_worker_replicas_for_trials(self, trials):
        # TODO: Improve provisioning algorithm
        return {
            trial : 2 # 2 replicas per trial
            for trial in trials
        }
    