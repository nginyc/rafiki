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
from typing import List, Type
import time
import traceback

from rafiki.utils.auth import superadmin_client
from rafiki.meta_store import MetaStore
from rafiki.model import load_model_class, BaseModel
from rafiki.advisor import Proposal
from rafiki.param_store import FileParamStore
from rafiki.predictor import Query, Prediction
from rafiki.redis import InferenceCache as RedisInferenceCache
from rafiki.kafka import InferenceCache as KafkaInferenceCache

LOOP_SLEEP_SECS = 0.1
PREDICT_BATCH_SIZE = 32

class InvalidWorkerError(Exception): pass
class InvalidTrialError(Exception): pass

logger = logging.getLogger(__name__)

class InferenceWorker():
    def __init__(self, service_id, worker_id, meta_store=None, param_store=None):
        self._service_id = service_id
        self._worker_id = worker_id
        self._meta_store = meta_store or MetaStore()
        self._param_store = param_store or FileParamStore()
        self._redis_host = os.environ['REDIS_HOST']
        self._redis_port = os.environ['REDIS_PORT']
        self._batch_size = PREDICT_BATCH_SIZE
        self._redis_cache: RedisInferenceCache = None
        self._inference_job_id = None
        self._model_inst: BaseModel = None
        self._proposal: Proposal = None
        self._store_params_id = None
        self._py_model_class: Type[BaseModel] = None
        self._kafka_cache = KafkaInferenceCache()

    def start(self):
        self._pull_job_info()
        self._redis_cache = RedisInferenceCache(self._inference_job_id, 
                                            self._redis_host, 
                                            self._redis_port)

        logger.info(f'Starting worker for inference job "{self._inference_job_id}"...')
        
        self._notify_start()

        # Load trial's model instance
        self._model_inst = self._load_trial_model()

        while True:
            queries = self._fetch_queries()
            if len(queries) > 0:
                predictions = self._predict(queries)
                self._submit_predictions(predictions)
            else:
                time.sleep(LOOP_SLEEP_SECS)

    def stop(self):
        self._notify_stop()

        # Run model destroy
        try:
            if self._model_inst is not None:
                self._model_inst.destroy()
        except:
            logger.error('Error destroying model:')
            logger.error(traceback.format_exc())

        # Run model class teardown
        try:
            if self._py_model_class is not None:
                self._py_model_class.teardown()
        except:
            logger.error('Error tearing down model class:')
            logger.error(traceback.format_exc())

    def _pull_job_info(self):
        service_id = self._service_id

        logger.info('Reading job info from meta store...')
        with self._meta_store:
            worker = self._meta_store.get_inference_job_worker(service_id)
            if worker is None:
                raise InvalidWorkerError('No such worker "{}"'.format(service_id))

            inference_job = self._meta_store.get_inference_job(worker.inference_job_id)
            if inference_job is None:
                raise InvalidWorkerError('No such inference job with ID "{}"'.format(worker.inference_job_id))

            trial = self._meta_store.get_trial(worker.trial_id)
            if trial is None or trial.store_params_id is None: # Must have model saved
                raise InvalidTrialError('No saved trial with ID "{}"'.format(worker.trial_id))
            logger.info(f'Using trial "{trial.id}"...')
            
            model = self._meta_store.get_model(trial.model_id)
            if model is None:
                raise InvalidTrialError('No such model with ID "{}"'.format(trial.model_id))
            logger.info(f'Using model "{model.name}"...')

            self._inference_job_id = inference_job.id

            self._py_model_class = load_model_class(model.model_file_bytes, model.model_class)
            self._proposal = Proposal.from_jsonable(trial.proposal)
            self._store_params_id = trial.store_params_id 

    def _load_trial_model(self):
        logger.info('Loading saved model parameters from store...')
        params = self._param_store.load(self._store_params_id)

        logger.info('Loading trial\'s trained model...')
        model_inst = self._py_model_class(**self._proposal.knobs)
        model_inst.load_parameters(params)

        return model_inst

    def _notify_start(self):
        superadmin_client().send_event('inference_job_worker_started', inference_job_id=self._inference_job_id)
        self._redis_cache.add_worker(self._worker_id)

    def _fetch_queries(self) -> List[Query]:
        queries = self._kafka_cache.pop_queries_for_worker(self._worker_id, self._batch_size)
        return queries

    def _predict(self, queries: List[Query]) -> List[Prediction]:
        # Pass queries to model, set null predictions if it errors
        try:
            predictions = self._model_inst.predict([x.query for x in queries])
        except:
            logger.error('Error while making predictions:')
            logger.error(traceback.format_exc())
            predictions = [None for x in range(len(queries))]

        # Transform predictions, adding associated worker & query ID
        predictions = [Prediction(x, query.id, self._worker_id) for (x, query) in zip(predictions, queries)]

        return predictions

    def _submit_predictions(self, predictions: List[Prediction]):
        self._kafka_cache.add_predictions_for_worker(self._worker_id, predictions)

    def _notify_stop(self):
        self._redis_cache.delete_worker(self._worker_id)
        superadmin_client().send_event('inference_job_worker_stopped', inference_job_id=self._inference_job_id)