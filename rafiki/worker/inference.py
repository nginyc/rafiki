import time
import uuid
import random
import os
import pickle
import logging
import traceback
import json

from rafiki.model import load_model_class
from rafiki.meta_store import MetaStore
from rafiki.param_store import ParamStore
from rafiki.cache import Cache
from rafiki.config import INFERENCE_WORKER_SLEEP, INFERENCE_WORKER_PREDICT_BATCH_SIZE

logger = logging.getLogger(__name__)

class InvalidWorkerException(Exception): pass

class InferenceWorker(object):
    def __init__(self, service_id, cache=None, meta_store=None, param_store=None):
        if cache is None: 
            cache = Cache()

        if meta_store is None: 
            meta_store = MetaStore()

        if param_store is None: 
            param_store = ParamStore()

        self._cache = cache
        self._meta_store = meta_store
        self._param_store = param_store
        self._service_id = service_id
        self._model = None
        
    def start(self):
        logger.info('Starting inference worker for service of id {}...' \
            .format(self._service_id))
        
        with self._meta_store:
            (inference_job_id, trial_id) = self._read_worker_info()

            # Add to inference job's set of running workers
            self._cache.add_worker_of_inference_job(self._service_id, inference_job_id)

            self._model = self._load_model(trial_id)

        while True:
            (query_ids, queries) = \
                self._cache.pop_queries_of_worker(self._service_id, INFERENCE_WORKER_PREDICT_BATCH_SIZE)
            
            if len(queries) > 0:
                logger.info('Making predictions for queries...')
                logger.info(queries)

                predictions = None
                try:
                    predictions = self._model.predict(queries)
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
        with self._meta_store:
            (inference_job_id, _) = self._read_worker_info()

        # Remove from inference job's set of running workers
        self._cache.delete_worker_of_inference_job(self._service_id, inference_job_id)

        if self._model is not None:
            self._model.destroy()
            self._model = None

    def _load_model(self, trial_id):
        trial = self._meta_store.get_trial(trial_id)
        sub_train_job = self._meta_store.get_sub_train_job(trial.sub_train_job_id)
        model = self._meta_store.get_model(sub_train_job.model_id)

        # Load model based on trial
        clazz = load_model_class(model.model_file_bytes, model.model_class)
        model_inst = clazz(**trial.knobs)

        # Load parameters from store, unpickle and load it in model
        parameters = self._param_store.get_params(trial.param_id)
        parameters = pickle.loads(parameters)
        model_inst.load_parameters(parameters)

        return model_inst

    def _read_worker_info(self):
        worker = self._meta_store.get_inference_job_worker(self._service_id)
        inference_job = self._meta_store.get_inference_job(worker.inference_job_id)

        if worker is None:
            raise InvalidWorkerException()

        return (
            inference_job.id,
            worker.trial_id
        )
