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

import os
from kafka import KafkaConsumer
from kafka import KafkaProducer 
from kafka.errors import KafkaError
from kafka import TopicPartition
import pickle
import logging
from typing import Union, List

from rafiki.predictor import Prediction, Query

logger = logging.getLogger(__name__)

RUNNING_INFERENCE_WORKERS = 'INFERENCE_WORKERS'
QUERIES_QUEUE = 'QUERIES'
PREDICTIONS_QUEUE = 'PREDICTIONS'

class InferenceCache(object):
    '''
    Caches queries & predictions to facilitate communication between predictor & inference workers.
    '''

    def __init__(self, hosts=os.environ.get('KAFKA_HOST', 'localhost'), ports=os.environ.get('KAFKA_PORT', 9092)):
        hostlist = hosts.split(',')
        portlist = ports.split(',')
        self.connection_url = [f'{host}:{port}' for host, port in zip(hostlist, portlist)]
        self.producer = KafkaProducer(bootstrap_servers=self.connection_url)   

    def add_predictions_for_worker(self, worker_id: str, predictions: List[Prediction]):
        logger.info(f'Adding {len(predictions)} prediction(s) for worker "{worker_id}"')
        
        for prediction in predictions:
            name = f'workers_{worker_id}_{prediction.query_id}_prediction'
            prediction = pickle.dumps(prediction)
            self.producer.send(name, key=name.encode('utf-8'),value=prediction)
            self.producer.flush()

    def take_prediction_for_worker(self, worker_id: str, query_id: str) -> Union[Prediction, None]:
        name = f'workers_{worker_id}_{query_id}_prediction'

        prediction_consumer = KafkaConsumer(name, bootstrap_servers=self.connection_url, auto_offset_reset='earliest', group_id=PREDICTIONS_QUEUE)
        prediction = None
        try:
            prediction = next(prediction_consumer).value
            prediction_consumer.commit()
            prediction = pickle.loads(prediction)
        except KafkaError:
            pass
        prediction_consumer.close()
        logger.info(f'Took prediction for query "{query_id}" from worker "{worker_id}"')
        return prediction

    def add_queries_for_worker(self, worker_id: str, queries: List[Query]):
        name = f'workers_{worker_id}_queries'
        queries = [pickle.dumps(x) for x in queries]
        logger.info(f'Adding {len(queries)} querie(s) for worker "{worker_id}"...')
        for query in queries:
            self.producer.send(name, key=name.encode('utf-8'), value=query)
            self.producer.flush()

    def pop_queries_for_worker(self, worker_id: str, batch_size: int) -> List[Query]:
        name = f'workers_{worker_id}_queries'

        query_consumer = KafkaConsumer(name, bootstrap_servers=self.connection_url, auto_offset_reset='earliest', group_id=QUERIES_QUEUE)
        
        partition = TopicPartition(name, 0)
        partitiondic = query_consumer.end_offsets([partition])
        offsetend = partitiondic.get(partition, None)
        if offsetend == 0:
            query_consumer.close()
            return []
        try:
            queries = []
            while True:
                record = next(query_consumer)
                queries.append(record.value)
                query_consumer.commit()
                if record.offset >= offsetend-1 or len(queries) == batch_size:
                    break
                
            queries = [pickle.loads(x) for x in queries]
            query_consumer.close()
            return queries
        except KafkaError:
            query_consumer.close()
            return []

    def __del__(self):
        self.producer.close()

def testquery():
    batch_size = 10
    worker_id = 10001
    op = RQueue(host='172.17.0.3', port=9092)
    queries = [i for i in range(batch_size)]
    op.add_queries_for_worker(worker_id, queries)
    
    popqueries = op.pop_queries_for_worker(worker_id, batch_size+10)
    print(popqueries)

def testprediction():
    worker_id = 358
    op = RQueue(host='172.17.0.3', port=9092)
    predictions = []
    for i in range(5):
        query_id = str(i)
        pre = Prediction(worker_id=worker_id, query_id=query_id, prediction=i)
        predictions.append(pre)

    op.add_predictions_for_worker(worker_id, predictions)
    for i in range(5):
        print(op.take_prediction_for_worker(worker_id, i))

if __name__ == '__main__':
    testquery()
    #testprediction()