import traceback
import logging
import os
import time

from db import Database

from ..containers import DockerSwarmContainerManager
from .provisioning import compute_train_worker_replicas_for_models

logger = logging.getLogger(__name__)

class Deployer(object):

    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._container_manager = container_manager

    def destroy_train_workers_for_train_job(self, train_job_id):
        with self._db:
            workers = self._db.get_train_job_workers_by_train_job(train_job_id)
        
            for worker in workers:
                if worker.service_id is not None:
                    self._container_manager.destroy_service(worker.service_id)

                self._db.mark_train_job_worker_as_stopped(worker)
                self._db.commit()

    def redeploy_workers_for_train_job(self, train_job_id):
        with self._db:
            train_job = self._db.get_train_job(train_job_id)
            workers = self._db.get_train_job_workers_by_train_job(train_job.id)
            models = self._db.get_models_by_task(train_job.task)
            model_to_replicas = compute_train_worker_replicas_for_models(models)

            for (model, replicas) in model_to_replicas.items():
                worker = next((x for x in workers if x.model_id == model.id), None)
                image_name = model.docker_image_name

                if worker is None:
                    worker = self._db.create_train_job_worker(train_job.id, model.id)
                    self._db.commit()

                if worker.service_id is None:
                    service_id = self._create_service_for_worker(worker.id, image_name, replicas)
                    self._db.update_train_job_worker(worker, service_id=service_id, replicas=replicas)
                    self._db.commit()
                
                # If actual worker replicas do not match intended replicas, update service's replicas
                if worker.replicas != replicas:
                    self._container_manager.update_service(worker.service_id, replicas)
                    self._db.update_train_job_worker(worker, replicas=replicas)
                    self._db.commit()


    def _create_service_for_worker(self, worker_id, image_name, replicas):
        service_name = 'rafiki_worker_{}'.format(worker_id)
        environment_vars = {
            'POSTGRES_HOST': os.environ['POSTGRES_HOST'],
            'POSTGRES_PORT': os.environ['POSTGRES_PORT'],
            'POSTGRES_USER': os.environ['POSTGRES_USER'],
            'POSTGRES_DB': os.environ['POSTGRES_DB'],
            'POSTGRES_PASSWORD': os.environ['POSTGRES_PASSWORD'],
            'LOGS_FOLDER_PATH': os.environ['LOGS_FOLDER_PATH']
        }

        # Mount logs folder onto workers too
        logs_folder_path = os.environ['LOGS_FOLDER_PATH']
        mounts = {
            logs_folder_path: logs_folder_path
        }

        service_id = self._container_manager.create_service(
            service_name=service_name, 
            image_name=image_name, 
            replicas=replicas, 
            args=[worker_id], 
            environment_vars=environment_vars,
            mounts=mounts
        )
        return service_id






    