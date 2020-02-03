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

import pytest

from rafiki.kafka import InferenceCache as KafkaInferenceCache
from rafiki.predictor import Prediction, Query

class TestKafkaCache():    
    
    @pytest.fixture(scope='class', autouse=True)
    def queris(self):
        '''
        Initializes queris for testing
        '''
        return {
            'test-query-id-1': [Query('query-1-1'), Query('query-1-2')],
            'test-query-id-2': [Query('query-2-1'), Query('query-2-2')],
            'test-query-id-3': [Query('query-3-1'), Query('query-3-2')]
        }

    @pytest.fixture(scope='class', autouse=True)
    def predictions(self):
        '''
        Initializes queris for testing
        '''
        return {
            'prediction-worker-1': [Prediction('prediction-raw-1-1', 'prediction-query-1-1', 'prediction-worker-1')],
            'prediction-worker-2': [Prediction('prediction-raw-2-1', 'prediction-query-2-1', 'prediction-worker-2')],
            'prediction-worker-3': [Prediction('prediction-raw-3-1', 'prediction-query-3-1', 'prediction-worker-3')]
        }

    @pytest.fixture(scope='class', autouse=True)
    def stores(self):
        '''
        Initializes params stores for testing
        '''
        return KafkaInferenceCache()

    def test_queris(self, queris, stores):
        # Populate params
        for query in queris.items():
            stores.add_queries_for_worker(query[0], query[1])

        for query in queris.items():
            q = stores.pop_queries_for_worker(query[0], len(query[1]))
            assert q == query[1]
        
    def test_prediction(self, predictions, stores):
        # Populate params
        for prediction in predictions.items():
            stores.add_predictions_for_worker(prediction[0], prediction[1])

        for prediction in predictions.items():
            p = stores.take_prediction_for_worker(prediction[0], prediction[1][0].query_id)
            assert p == prediction[1][0]