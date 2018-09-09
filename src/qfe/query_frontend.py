from cache import Cache
from common import REQUEST_QUEUE, QFE_SLEEP
import uuid
import time
import json

class QueryFrontend(object):

    def __init__(self, cache=Cache()):
        self._cache = cache

    def predict(self, input):
        id = str(uuid.uuid4())
        request = {
            'id': id,
            'input': input
        }
        self._cache.append_list(REQUEST_QUEUE, request)
        
        while True:
            prediction = self._cache.get(id)

            if prediction is not None:
                return prediction.decode()
            else:
                time.sleep(QFE_SLEEP)
            