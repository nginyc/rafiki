import numpy as np
from db import Database
from cache import Cache
from common import RUNNING_INFERENCE_WORKERS, REQUEST_QUEUE, INFERENCE_WORKER_SLEEP, BATCH_SIZE
import time
import uuid
import random

# For testing only
class Model(object):

    def __init__(self): pass

    # queries is an [[int]], [[double]], [[string]]
    # returns a [[string]] where each [string] is a prediction
    def predict(self, queries):
        result = []
        for query in queries:
            digit = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8 ,9])
            result.append([str(digit)])
        return result

class InferenceWorker(object):

    #TODO: remove model_name
    def __init__(self, cache=Cache(), db=Database(), trial_id=1, model_name=os.getenv('MODEL_NAME')):
        self._cache = cache
        self._db = db
        self._load_model(trial_id, model_name)
        self._add_id()

    def _load_model(self, trial_id, model_name):
        #TODO: load model from db and unserialize it.
        # self._model = model (required)
        # self._model_name = model_name (required)
        self._model = Model()
        self._model_name = model_name

    def _add_id(self):
        self._id = str(uuid.uuid4())
        self._cache.append_list(RUNNING_INFERENCE_WORKERS, '{}_{}'.format(self._model_name, self._id))

    def start(self):
        queue_key = '{}_{}_{}'.format(REQUEST_QUEUE, self._model_name, self._id)
        while True:
            requests = self._cache.get_list_range(queue_key, 0, BATCH_SIZE - 1)
            ids = []
            inputs = None

            for request in requests:
                if inputs is None:
                    inputs = request['input']
                else:
                    inputs = np.vstack([inputs, request['input']])
                ids.append(request['id'])
            
            if len(ids) > 0:
                predictions = self._model.predict(inputs)
                for (id, prediction) in zip(ids, predictions):
                    self._cache.append_list(id, prediction)

            time.sleep(INFERENCE_WORKER_SLEEP)

