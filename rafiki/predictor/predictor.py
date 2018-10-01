import uuid
import time
import json
import logging

from rafiki.cache import Cache
from rafiki.db import Database
from rafiki.config import RUNNING_INFERENCE_WORKERS, REQUEST_QUEUE, QFE_SLEEP

logger = logging.getLogger(__name__)
 
class Predictor(object):

    def __init__(self, service_id, db=Database(), cache=Cache()):
        self._service_id = service_id
        self._db = db
        self._cache = cache

    def predict(self, query):
        logger.info('Received query:')
        logger.info(query)

        id = str(uuid.uuid4())
        request = {
            'id': id,
            'query': query
        }

        running_inference_workers = self._get_inference_workers()
        request_ids = set()
        for running_inference_worker in running_inference_workers:
            queue_key = '{}_{}'.format(REQUEST_QUEUE, running_inference_worker.decode())
            request_key = '{}_{}'.format(id, running_inference_worker.decode())
            self._cache.append_list(queue_key, request)
            request_ids.add(request_key)
        
        response_ids = set()
        responses = { 'responses': [] }

        logger.info('Waiting for predictions...')

        #TODO: add SLO. break loop when timer is out.
        while True:
            unresponded_ids = request_ids - response_ids
            for unresponded_id in unresponded_ids:
                prediction_jsons = self._cache.get_list_range(unresponded_id, 0, -1)
                if len(prediction_jsons) > 0:
                    prediction_json = prediction_jsons[0]
                    prediction = json.loads(prediction_json)
                    response_ids.add(unresponded_id)
                    responses['responses'].append(prediction)
                    self._cache.delete(unresponded_id)
            if (request_ids == response_ids): break
            time.sleep(QFE_SLEEP)

        logger.info('Responding with predictions:')
        logger.info(responses)

        return responses

    def _get_inference_workers(self):
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, self._service_id)
        return self._cache.get_set(inference_workers_key)

    def predict_batch(self, queries):
        #TODO: implement method
        pass