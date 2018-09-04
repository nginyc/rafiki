import traceback
import logging
import os
import time

from db import Database

from .containers import DockerSwarmContainerManager
from .provisioning import compute_train_worker_replicas_for_models
from .budget import if_train_job_budget_reached

logger = logging.getLogger(__name__)

class Daemon(object):
    DB_POLL_INTERVAL_SECONDS = 10

    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._container_manager = container_manager

    def start(self):
        with self._db:
            while True:
                logger.info('Reprovisioning train workers...')
                self._reprovision_train_workers()
                sleep_secs = self.DB_POLL_INTERVAL_SECONDS
                time.sleep(sleep_secs)
                logger.info('Sleeping for {}s...'.format(sleep_secs))
                    
    def _reprovision_train_workers(self):
        train_jobs = self._db.get_uncompleted_train_jobs()

        for train_job in train_jobs:
            completed_trials = self._db.get_completed_trials_by_train_job(train_job.id)

            if_budget_reached = if_train_job_budget_reached(
                budget_type=train_job.budget_type,
                budget_amount=train_job.budget_amount,
                completed_trials=completed_trials
            )
            
            if if_budget_reached:
                self._destroy_train_workers_for_train_job(train_job)
                self._db.mark_train_job_as_complete(train_job)
            else:
                self._redeploy_train_workers_for_train_job(train_job)

    def _destroy_train_workers_for_train_job(self, train_job):
        workers = self._db.get_train_job_workers_by_train_job(train_job.id)
        
        for worker in workers:

            if worker.service_id is not None:
                self._container_manager.destroy_service(worker.service_id)

            self._db.destroy_train_job_worker(worker.id)

    def _redeploy_train_workers_for_train_job(self, train_job):
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
                self._db.update_train_job_worker(worker, service_id=service_id)
                self._db.commit()
            
            # If actual worker replicas do not match intended replicas, update service's replicas
            if worker.replicas != replicas:
                self._container_manager.update_service(worker.service_id, replicas)
                self._db.update_train_job_worker(worker, replicas=replicas)
                self._db.commit()

    def _create_service_for_worker(self, worker_id, image_name, replicas):
        service_name = 'rafiki_worker_{}'.format(worker_id)
        environment_vars = {
            'POSTGRES_HOST': os.environ.get('POSTGRES_HOST', 'localhost'),
            'POSTGRES_PORT': os.environ.get('POSTGRES_PORT', 5432),
            'POSTGRES_USER': os.environ.get('POSTGRES_USER', 'rafiki'),
            'POSTGRES_DB': os.environ.get('POSTGRES_DB', 'rafiki'),
            'POSTGRES_PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'rafiki')
        }

        service_id = self._container_manager.create_service(
            service_name=service_name, 
            image_name=image_name, 
            replicas=replicas, 
            args=[worker_id], 
            environment_vars=environment_vars
        )
        return service_id






    