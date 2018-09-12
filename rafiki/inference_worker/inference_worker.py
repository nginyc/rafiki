import numpy as np
import time
import uuid
import random
import os
import ast

from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import RUNNING_INFERENCE_WORKERS, REQUEST_QUEUE, INFERENCE_WORKER_SLEEP, BATCH_SIZE

#TODO: For testing only. remove when not needed.
class Model(object):

    def __init__(self): pass

    # queries is an [[int]], [[double]], [[string]]
    # returns a [[string]] where each [string] is a prediction
    def predict(self, queries):
        result = []
        for query in queries:
            digit1 = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8 ,9])
            digit2 = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8 ,9])
            result.append([str(digit1), str(digit2)])
        return result

class InferenceWorker(object):

    def __init__(self, 
        cache=Cache(), 
        db=None, 
        inference_job_id=os.getenv('INFERENCE_JOB_ID'),
        trial_id=os.getenv('TRIAL_ID'), 
        model_name=os.getenv('MODEL_NAME')):
        
        self._cache = cache
        self._db = db #TODO change to Database()
        self._inference_job_id = inference_job_id
        self._trial_id = trial_id
        self._model_name = model_name
        self._load_model(trial_id)
        self._add_id()

    def _load_model(self, trial_id):
        #TODO: load model from rafiki.db and unserialize it.
        # self._model = model (required)
        self._model = Model()

    def _add_id(self):
        self._id = str(uuid.uuid4())
        self._worker_id = '{}_{}_{}_{}'.format(
            self._inference_job_id,
            self._trial_id,
            self._model_name,
            self._id
        )
        self._cache.append_list(RUNNING_INFERENCE_WORKERS, self._worker_id)

    def start(self):
        queue_key = '{}_{}'.format(REQUEST_QUEUE, self._worker_id)
        while True:
            requests = self._cache.get_list_range(queue_key, 0, BATCH_SIZE - 1)
            ids = []
            queries = None

            for request in requests:
                request = ast.literal_eval(request.decode())
                if queries is None:
                    queries = request['query']
                else:
                    queries = np.vstack([queries, request['query']])
                ids.append(request['id'])
            
            if len(ids) > 0:
                predictions = self._model.predict(queries)
                self._cache.trim_list(queue_key, len(ids), -1)
                for (id, prediction) in zip(ids, predictions):
                    request_id = '{}_{}'.format(id, self._worker_id)
                    self._cache.append_list_seq(request_id, prediction)
            time.sleep(INFERENCE_WORKER_SLEEP)

    def stop(self):
        queue_key = '{}_{}'.format(REQUEST_QUEUE, self._worker_id)
        self._cache.delete_list_value(RUNNING_INFERENCE_WORKERS, self._worker_id)
        self._cache.delete(queue_key)
