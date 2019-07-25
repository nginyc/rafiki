#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from typing import Union, List
import pickle
import logging

from rafiki.predictor import Prediction, Query
from .redis import RedisSession

logger = logging.getLogger(__name__)

REDIS_NAMESPACE = 'INFERENCE'

class InferenceCache():
    '''
    Caches queries & predictions to facilitate communication between predictor & inference workers.

    For each session, assume a single predictor and multiple inference workers running concurrently.

    :param str session_id: Associated session ID
    '''

    '''
        Internally, organises data into these Redis namespaces:

        workers                                     | Set of IDs of workers available 
        workers:<worker_id>:queries                 | List of queries for worker
        workers:<worker_id>:<query_id>:prediction   | Prediction for query of ID for worker of ID 
    '''
    
    def __init__(self, 
                session_id='local', 
                redis_host=None,
                redis_port=None):
        redis_namespace = f'{REDIS_NAMESPACE}:{session_id}'
        self._redis = RedisSession(redis_namespace, redis_host, redis_port)

    ####################################
    # Predictor
    ####################################

    def get_workers(self) -> List[str]:
        worker_ids = self._redis.list_set('workers')
        return worker_ids

    def add_queries_for_worker(self, worker_id: str, queries: List[Query]):
        name = f'workers:{worker_id}:queries'
        queries = [pickle.dumps(x) for x in queries]
        logger.info(f'Adding {len(queries)} querie(s) for worker "{worker_id}"...')
        self._redis.prepend_to_list(name, *queries)

    def take_prediction_for_worker(self, worker_id: str, query_id: str) -> Union[Prediction, None]:
        name = f'workers:{worker_id}:{query_id}:prediction'
        prediction = self._redis.get(name)
        if prediction is None:
            return None
        
        # Delete prediction from cache
        self._redis.delete(name)
        prediction = pickle.loads(prediction)
        logger.info(f'Took prediction for query "{query_id}" from worker "{worker_id}"')
        return prediction

    def clear_all(self):
        self._redis.delete('workers')
        self._redis.delete_pattern('workers:*')
        self._redis.delete_pattern('queries:*')

    ####################################
    # Inference Worker
    ####################################

    def add_worker(self, worker_id: str):
        self._redis.add_to_set('workers', worker_id)

    def pop_queries_for_worker(self, worker_id: str, batch_size: int) -> List[Query]:
        name = f'workers:{worker_id}:queries'
        queries = []

        # Repeatedly pop from list of queries to accumulate batch
        for _ in range(batch_size):
            query = self._redis.pop_from_list(name)
            if query is None:
                break
            query = pickle.loads(query)
            queries.append(query)
        
        if len(queries) > 0:
            logger.info(f'Popped {len(queries)} querie(s) for worker "{worker_id}"')

        return queries
    
    def add_predictions_for_worker(self, worker_id: str, predictions: List[Prediction]):
        logger.info(f'Adding {len(predictions)} prediction(s) for worker "{worker_id}"')
        for prediction in predictions:
            name = f'workers:{worker_id}:{prediction.query_id}:prediction'
            prediction = pickle.dumps(prediction)
            self._redis.set(name, prediction)
    

    def delete_worker(self, worker_id: str):
        self._redis.delete_from_set('workers', worker_id)
