import time
import json
import logging
import pickle

from rafiki.cache import Cache
from rafiki.db import Database
from rafiki.config import PREDICTOR_PREDICT_SLEEP

from .ensemble import ensemble_predictions

logger = logging.getLogger(__name__)
 
class Predictor(object):

    def __init__(self, service_id, db=None, cache=None):
        if db is None: 
            db = Database()
        if cache is None: 
            cache = Cache()

        self._service_id = service_id
        self._db = db
        self._cache = cache

    def start(self):
        with self._db:
            (self._inference_job_id, self._task) \
                = self._read_predictor_info()

    def predict(self, query):
        logger.info('Received query:')
        logger.info(query)

        running_worker_ids = self._cache.get_workers_of_inference_job(self._inference_job_id)
        worker_to_prediction = {}
        worker_to_query_id = {}
        responded_worker_ids = set() 
        for worker_id in running_worker_ids:
            query_id = self._cache.add_query_of_worker(worker_id, query)
            worker_to_query_id[worker_id] = query_id

        logger.info('Waiting for predictions from workers...')

        #TODO: add SLO. break loop when timer is out.
        while True:
            for (worker_id, query_id) in worker_to_query_id.items():
                if worker_id in responded_worker_ids:
                    continue
                    
                prediction = self._cache.pop_prediction_of_worker(worker_id, query_id)
                if prediction is not None:
                    worker_to_prediction[worker_id] = prediction
                    responded_worker_ids.add(worker_id)
             
            if len(responded_worker_ids) == len(running_worker_ids): 
                break

            time.sleep(PREDICTOR_PREDICT_SLEEP)

        logger.info('Predictions:')
        logger.info(worker_to_prediction)

        predictions_list = [
            [worker_to_prediction[worker_id]]
            for worker_id in running_worker_ids
        ]

        predictions = ensemble_predictions(predictions_list, self._task)
        prediction = predictions[0] if len(predictions) > 0 else None

        return {
            'prediction': prediction
        }

    def _read_predictor_info(self):
        inference_job = self._db.get_inference_job_by_predictor(self._service_id)
        train_job = self._db.get_train_job(inference_job.train_job_id)

        return (
            inference_job.id,
            train_job.task
        )

    def predict_batch(self, queries):
        #TODO: implement method
        pass
