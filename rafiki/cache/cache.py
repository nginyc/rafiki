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

import redis
import os
import json
import uuid

RUNNING_INFERENCE_WORKERS = 'INFERENCE_WORKERS'
QUERIES_QUEUE = 'QUERIES'
PREDICTIONS_QUEUE = 'PREDICTIONS'

class Cache(object):
    def __init__(self,
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=os.environ.get('REDIS_PORT', 6379)):

        cache_connection_url = self._make_connection_url(
            host=host,
            port=port
        )

        self._connection_pool = redis.ConnectionPool.from_url(cache_connection_url)
        self._redis = redis.StrictRedis(connection_pool=self._connection_pool, decode_responses=True)
        
    def add_worker_of_inference_job(self, worker_id, inference_job_id):
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, inference_job_id)
        self._redis.sadd(inference_workers_key, worker_id)

    def delete_worker_of_inference_job(self, worker_id, inference_job_id):
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, inference_job_id)
        self._redis.srem(inference_workers_key, worker_id)

    def get_workers_of_inference_job(self, inference_job_id):
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, inference_job_id)
        worker_ids = self._redis.smembers(inference_workers_key)
        return [x.decode() for x in worker_ids]

    def add_query_of_worker(self, worker_id, query):
        query_id = str(uuid.uuid4())
        query = json.dumps({
            'id': query_id,
            'query': query
        })

        worker_queries_key = '{}_{}'.format(QUERIES_QUEUE, worker_id)
        self._redis.rpush(worker_queries_key, query)
        return query_id

    def pop_queries_of_worker(self, worker_id, batch_size):
        worker_queries_key = '{}_{}'.format(QUERIES_QUEUE, worker_id)
        queries = self._redis.lrange(worker_queries_key, 0, batch_size - 1)
        self._redis.ltrim(worker_queries_key, len(queries), -1)
        queries = [json.loads(x) for x in queries]
        query_ids = [x['id'] for x in queries]
        queries = [x['query'] for x in queries]
        return (query_ids, queries)

    def add_prediction_of_worker(self, worker_id, query_id, prediction):
        prediction = json.dumps({
            'id': query_id,
            'prediction': prediction
        })

        worker_predictions_key = '{}_{}'.format(PREDICTIONS_QUEUE, worker_id)
        self._redis.rpush(worker_predictions_key, prediction)

    def pop_prediction_of_worker(self, worker_id, query_id):
        # Search through worker's list of predictions
        worker_predictions_key = '{}_{}'.format(PREDICTIONS_QUEUE, worker_id)
        predictions = self._redis.lrange(worker_predictions_key, 0, -1)

        for (i, prediction) in enumerate(predictions):
            prediction = json.loads(prediction)
            # If prediction for query found, remove prediction from list and return it
            if prediction['id'] == query_id:
                self._redis.ltrim(worker_predictions_key, i + 1, i)
                return prediction['prediction']

        # Return None if prediction is not found
        return None

    def _make_connection_url(self, host, port):
        return 'redis://{}:{}'.format(host, port)
