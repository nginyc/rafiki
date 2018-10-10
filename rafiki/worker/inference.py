import time
import uuid
import random
import os
import numpy as np
import logging
import traceback
import json

from rafiki.utils.model import load_model_class, parse_model_prediction
from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import INFERENCE_WORKER_SLEEP, INFERENCE_WORKER_PREDICT_BATCH_SIZE

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
            (inference_job_id, trial_id) = self._read_worker_info()
            self._model = self._load_model(trial_id)

        # Add to inference job's set of running workers
        self._cache.add_worker_of_inference_job(self._service_id, inference_job_id)
            
        while True:
            (query_ids, queries) = \
                self._cache.pop_queries_of_worker(self._service_id, INFERENCE_WORKER_PREDICT_BATCH_SIZE)
            
            if len(queries) > 0:
                logger.info('Making predictions for queries...')
                logger.info(queries)

                predictions = None
                try:
                    predictions = self._model.predict(queries)
                    predictions = [parse_model_prediction(x) for x in predictions]
                except Exception:
                    logger.error('Error while making predictions:')
                    logger.error(traceback.format_exc())
                    
                if predictions is not None:
                    logger.info('Predictions:')
                    logger.info(predictions)

                    for (query_id, prediction) in zip(query_ids, predictions):
                        self._cache.add_prediction_of_worker(self._service_id, query_id, prediction)

            time.sleep(INFERENCE_WORKER_SLEEP)

    def stop(self):
        with self._db:
            (inference_job_id, _) = self._read_worker_info()

        # Remove from inference job's set of running workers
        self._cache.delete_worker_of_inference_job(self._service_id, inference_job_id)

        if self._model is not None:
            self._model.destroy()
            self._model = None

    def _load_model(self, trial_id):
        trial = self._db.get_trial(trial_id)
        model = self._db.get_model(trial.model_id)

        # Load model based on trial
        clazz = load_model_class(model.model_file_bytes, model.model_class)
        model_inst = clazz()
        model_inst.init(trial.knobs)
        model_inst.load_parameters(trial.parameters)

        return model_inst

    def _read_worker_info(self):
        worker = self._db.get_inference_job_worker(self._service_id)
        inference_job = self._db.get_inference_job(worker.inference_job_id)

        if worker is None:
            raise InvalidWorkerException()

        return (
            inference_job.id,
            worker.trial_id
        )
