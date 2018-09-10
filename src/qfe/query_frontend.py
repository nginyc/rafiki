from cache import Cache
from config import RUNNING_INFERENCE_WORKERS, REQUEST_QUEUE, QFE_SLEEP
import uuid
import time
import json

class QueryFrontend(object):

    def __init__(self, cache=Cache()):
        self._cache = cache

    def predict(self, query):
        id = str(uuid.uuid4())
        request = {
            'id': id,
            'query': query
        }
        running_inference_workers = self._cache.get_list_range(RUNNING_INFERENCE_WORKERS, 0, -1)
        request_ids = set()
        for running_inference_worker in running_inference_workers:
            queue_key = '{}_{}'.format(REQUEST_QUEUE, running_inference_worker)
            request_key = '{}_{}'.format(id, running_inference_worker)
            self._cache.append_list(queue_key, request)
            request_ids.add(request_key)
        
        response_ids = set()
        responses = { 'responses': [] }
        while True:
            unresponded_ids = request_ids - response_ids
            for unresponded_id in unresponded_ids:
                prediction = self._cache.get(unresponded_id)
                if prediction is not None:
                    response_ids.add(unresponded_id)
                    keys = unresponded_id.split('_')
                    prediction = { 
                        'inference_job_id': keys[1],
                        'trial_id': keys[2]
                        'model_name': keys[3],
                        'inference_worker_id': keys[4],
                        'prediction': prediction
                    }
                    responses['responses'].append(prediction)
                    self._cache.delete(unresponded_id)
            if (request_ids == response_ids): break
            time.sleep(QFE_SLEEP)
        return responses

    def predict_batch(self, queries):
        #TODO: implement method