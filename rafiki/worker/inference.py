import time
import uuid
import random
import os
import numpy as np
import ast
import logging
import traceback
import json

from rafiki.utils.model import load_model_class
from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import RUNNING_INFERENCE_WORKERS, REQUEST_QUEUE, INFERENCE_WORKER_SLEEP, BATCH_SIZE

logger = logging.getLogger(__name__)

class InvalidWorkerException(Exception):
    pass

class InferenceWorker(object):
    def __init__(self, service_id, cache=Cache(), db=Database()):
        self._cache = cache
        self._db = db
        self._service_id = service_id
        self._model = None
        
    def start(self):
        logger.info('Starting inference worker for service of id {}...' \
            .format(self._service_id))
        
        with self._db:
            (predictor_service_id, trial_id) = self._read_worker_info()
            self._model = self._load_model(trial_id)

        self._register_worker(predictor_service_id)
            
        queue_key = self._get_queue_key()
        while True:
            requests = self._cache.get_list_range(queue_key, 0, BATCH_SIZE - 1)
            ids = []
            queries = []

            for request in requests:
                request = ast.literal_eval(request.decode())
                queries.append(request['query'])
                ids.append(request['id'])
            
            if len(ids) > 0:
                logger.info('Making predictions for queries...')
                logger.info(queries)

                predictions = None
                try:
                    predictions = self._model.predict(queries)
                    predictions = [to_json_serializable(x) for x in predictions]
                except Exception:
                    logger.error('Error while making predictions:')
                    logger.error(traceback.format_exc())
                    
                if predictions is not None:
                    logger.info('Predictions:')
                    logger.info(predictions)

                    self._cache.trim_list(queue_key, len(ids), -1)
                    for (id, prediction) in zip(ids, predictions):
                        prediction_json = json.dumps(prediction)
                        request_id = '{}_{}'.format(id, self._service_id)
                        self._cache.append_list(request_id, prediction_json)

            time.sleep(INFERENCE_WORKER_SLEEP)

    def stop(self):
        with self._db:
            (predictor_service_id, _) = self._read_worker_info()

        self._unregister_worker(predictor_service_id)

        if self._model is not None:
            self._model.destroy()
            self._model = None

    def _get_queue_key(self):
        return '{}_{}'.format(REQUEST_QUEUE, self._service_id)

    def _load_model(self, trial_id):
        trial = self._db.get_trial(trial_id)
        model = self._db.get_model(trial.model_id)

        # Load model based on trial
        clazz = load_model_class(model.model_file_bytes, model.model_class)
        model_inst = clazz()
        model_inst.init(trial.knobs)
        model_inst.load_parameters(trial.parameters)

        return model_inst

    def _unregister_worker(self, predictor_service_id):
        # Remove from predictor's set of running workers
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, predictor_service_id)
        self._cache.remove_from_set(inference_workers_key, self._service_id)

        # Remove own queries queue
        queue_key = self._get_queue_key()
        self._cache.delete(queue_key)

    def _register_worker(self, predictor_service_id):
        # Add to predictor's set of running workers
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, predictor_service_id)
        self._cache.add_to_set(inference_workers_key, self._service_id)

    def _read_worker_info(self):
        worker = self._db.get_inference_job_worker(self._service_id)
        inference_job = self._db.get_inference_job(worker.inference_job_id)

        if worker is None:
            raise InvalidWorkerException()

        return (
            inference_job.predictor_service_id,
            worker.trial_id
        )

def to_json_serializable(data):
    if isinstance(data, np.int64):
        return int(data)

    return data