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

import logging
import os
from collections import defaultdict
from typing import List, Callable, Any
import time
import traceback

from rafiki.utils.auth import superadmin_client
from rafiki.meta_store import MetaStore
from rafiki.redis import InferenceCache as RedisInferenceCache
from rafiki.kafka import InferenceCache as KafkaInferenceCache

from .constants import Prediction, Query
from .ensemble import get_ensemble_method

PREDICT_LOOP_SLEEP_SECS = 0.1

class InvalidInferenceJobError(Exception): pass

logger = logging.getLogger(__name__)

class Predictor():
    def __init__(self, service_id, meta_store=None):
        self._service_id = service_id
        self._meta_store = meta_store or MetaStore()
        self._redis_host = os.getenv('REDIS_HOST', 'rafiki_redis')
        self._redis_port = os.getenv('REDIS_PORT', 6379)
        self._kafka_host = os.getenv('KAFKA_HOST', 'rafiki_kafka')
        self._kafka_port = os.getenv('KAFKA_PORT', 9092)
        self._ensemble_method: Callable[[List[Any]], Any] = None
        self._inference_job_id = None

        self._pull_job_info()
        self._redis_cache = RedisInferenceCache(self._inference_job_id, 
                                                self._redis_host, 
                                                self._redis_port)
        self._kakfa_cache = KafkaInferenceCache()
        logger.info(f'Initialized predictor for inference job "{self._inference_job_id}"')

    # Only a single thread should run this
    def start(self):
        self._notify_start()

    def predict(self, queries):
        worker_predictions_list = self._get_predictions_from_workers(queries)
        predictions = self._combine_worker_predictions(worker_predictions_list)
        return predictions

    # Only a single thread should run this
    def stop(self):
        self._notify_stop()

        # Clear caches for inference job
        try:
            self._redis_cache.clear_all()
        except:
            logger.error('Error clearing inference cache:')
            logger.error(traceback.format_exc())

    def _pull_job_info(self):
        service_id = self._service_id

        logger.info('Reading job info from meta store...')
        with self._meta_store:
            inference_job = self._meta_store.get_inference_job_by_predictor(service_id)
            if inference_job is None:
                raise InvalidInferenceJobError('No inference job associated with predictor "{}"'.format(service_id))

            train_job = self._meta_store.get_train_job(inference_job.train_job_id)
            if train_job is None:
                raise InvalidInferenceJobError('No such train job with ID "{}"'.format(inference_job.train_job_id))

            self._ensemble_method = get_ensemble_method(train_job.task)
            self._inference_job_id = inference_job.id

            logger.info(f'Using ensemble method: {self._ensemble_method}...')

    def _notify_start(self):
        superadmin_client().send_event('predictor_started', inference_job_id=self._inference_job_id)

    def _get_predictions_from_workers(self, queries: List[Any]) -> List[List[Prediction]]:
        queries = [Query(x) for x in queries]

        # Wait for at least 1 free worker
        worker_ids = []
        while len(worker_ids) == 0:
            worker_ids = self._redis_cache.get_workers()

        # For each worker, send queries to worker
        pending_queries = set() # {(query_id, worker_id)}
        for worker_id in worker_ids:
            self._kakfa_cache.add_queries_for_worker(worker_id, queries)
            # self._redis_cache.add_queries_for_worker(worker_id, queries)
            pending_queries.update([(x.id, worker_id) for x in queries])

        # Wait for all predictions to be made
        query_id_to_predictions = defaultdict(list) # { <query_id>: [prediction] }
        while len(pending_queries) > 0:
            # For every pending query to worker
            for (query_id, worker_id) in list(pending_queries):
                # Check cache
                prediction = self._kakfa_cache.take_prediction_for_worker(worker_id, query_id)
                # prediction = self._redis_cache.take_prediction_for_worker(worker_id, query_id)
                if prediction is None:
                    continue
                
                # Record prediction & mark as not pending
                query_id_to_predictions[query_id].append(prediction)
                pending_queries.remove((query_id, worker_id))
            
            time.sleep(PREDICT_LOOP_SLEEP_SECS)

        # Reorganize predictions
        worker_predictions_list = []
        for query in queries:
            worker_predictions = query_id_to_predictions[query.id]
            worker_predictions_list.append(worker_predictions)

        return worker_predictions_list

    def _combine_worker_predictions(self, worker_predictions_list: List[List[Prediction]]) -> List[Any]:
        # Ensemble predictions for each query
        predictions = []
        for worker_predictions in worker_predictions_list:
            # Transform predictions & remove all null predictions
            worker_predictions = [x.prediction for x in worker_predictions if x.prediction is not None] 

            # Do ensembling
            prediction = self._ensemble_method(worker_predictions)
            predictions.append(prediction)

        return predictions

    def _notify_stop(self):
        superadmin_client().send_event('predictor_stopped', inference_job_id=self._inference_job_id)