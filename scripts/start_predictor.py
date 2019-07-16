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
import logging

from rafiki.utils.service import run_worker
from rafiki.meta_store import MetaStore
from rafiki.predictor.predictor import Predictor
from rafiki.predictor.app import app

logger = logging.getLogger(__name__)

global_predictor: Predictor = None

def start_worker(service_id, service_type, container_id):
    global global_predictor
    
    logger.info('Starting global predictor...')
    global_predictor = Predictor(service_id)
    global_predictor.start()

    app.run(host='0.0.0.0', 
            port=os.getenv('PREDICTOR_PORT', 3003), 
            threaded=True)

def stop_worker():
    global global_predictor

    if global_predictor is not None:
        global_predictor.stop()
        global_predictor = None

meta_store = MetaStore()
run_worker(meta_store, start_worker, stop_worker)
