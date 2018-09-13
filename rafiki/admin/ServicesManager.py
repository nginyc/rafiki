import numpy as np
import os
import logging
import traceback

from rafiki.db import Database
from rafiki.model import unserialize_model, serialize_model
from rafiki.constants import ServiceStatus, UserType, ServiceType
from rafiki.config import BASE_MODEL_IMAGE, QUERY_FRONTEND_IMAGE, \
    MIN_SERVICE_PORT, MAX_SERVICE_PORT, QUERY_FRONTEND_PORT

from .train import compute_train_worker_replicas_for_models
from .containers import DockerSwarmContainerManager 
from .auth import hash_password, if_hash_matches_password

logger = logging.getLogger(__name__)

class UserExistsException(Exception):
    pass

class NoSuchUserException(Exception): 
    pass

class InvalidPasswordException(Exception):
    pass

class ServicesManager(object):
    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._container_manager = container_manager

    def create_inference_job_services(self, inference_job_id):
        inference_job = self._db.get_inference_job(inference_job_id)
        query_service = self._create_query_service(inference_job)

        # TODO: Deploy inference workers
        # best_trials = self._db.get_best_trials_of_app(app, max_count=max_models)
        # best_trials_models = [self._db.get_model(x.model_id) for x in best_trials]

        # inference_job = self._db.create_inference_job(user_id, app)
        # self._db.commit() 

        self._db.mark_inference_job_as_running(
            inference_job,
            query_service_id=query_service.id
        )
        self._db.commit()
        return (inference_job, query_service)

    def stop_inference_job_services(self, inference_job_id):
        inference_job = self._db.get_inference_job(inference_job_id)
        # TODO: Destroy inference workers
        service = self._db.get_service(inference_job.query_service_id)
        self._destroy_service(service)
        self._db.mark_inference_job_as_stopped(inference_job)
        self._db.commit()
        return inference_job

    def create_train_job_services(self, train_job_id):
        train_job = self._db.get_train_job(train_job_id)
        models = self._db.get_models_of_task(train_job.task)
        model_to_replicas = compute_train_worker_replicas_for_models(models)

        for (model, replicas) in model_to_replicas.items():
            # Create corresponding service for newly created worker
            self._create_train_job_service(train_job, model, replicas)

        self._update_train_job_status(train_job)
        return train_job

    def stop_train_job_services(self, train_job_id):
        train_job = self._db.get_train_job(train_job_id)
        # Stop all services for train job
        train_job_services = self._db.get_services_of_train_job(train_job_id)
        for train_job_service in train_job_services:
            self._stop_train_job_service(train_job_service)

        return train_job
        
    def stop_train_job_service(self, train_job_service_id):
        train_job_service = self._db.get_train_job_service(train_job_service_id)
        self._stop_train_job_service(train_job_service)
        return train_job_service

    ####################################
    # Private
    ####################################

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
            docker_image=QUERY_FRONTEND_IMAGE,
            replicas=1,
            environment_vars=environment_vars,
            container_port=QUERY_FRONTEND_PORT
        )

        return service

    def _create_train_job_service(self, train_job, model, replicas):
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
            'SUPERADMIN_EMAIL': os.environ['SUPERADMIN_EMAIL'],
            'SUPERADMIN_PASSWORD': os.environ['SUPERADMIN_PASSWORD']
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=model.docker_image,
            replicas=replicas,
            environment_vars=environment_vars
        )

        self._db.create_train_job_service(
            service_id=service.id,
            train_job_id=train_job.id,
            model_id=model.id
        )

        return service

    def _stop_train_job_service(self, train_job_service):
        service = self._db.get_service(train_job_service.service_id)
        self._destroy_service(service)
        train_job = self._db.get_train_job(train_job_service.train_job_id)
        self._update_train_job_status(train_job)

    def _update_train_job_status(self, train_job):
        train_job_services = self._db.get_services_of_train_job(train_job.id)
        services = [self._db.get_service(x.service_id) for x in train_job_services]
        
        # If all services for the train job have stopped, stop train job as well
        if next((
            x for x in services 
            if x.status in [ServiceStatus.RUNNING, ServiceStatus.STARTED]
        ), None) is None:
            self._db.mark_train_job_as_complete(train_job)
            self._db.commit()

    def _destroy_service(self, service):
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
            ext_port = self._get_available_service_port()
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

    # Compute next available port
    def _get_available_service_port(self):
        services = self._db.get_services(status=ServiceStatus.RUNNING)
        used_ports = [x.port for x in services]
        port = MIN_SERVICE_PORT
        while port <= MAX_SERVICE_PORT:
            if port not in used_ports:
                return port

            port += 1

        return port
    