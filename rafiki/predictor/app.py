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
from flask import Flask, jsonify, request, g
import threading

from .predictor import Predictor

service_id = os.environ['RAFIKI_SERVICE_ID']

app = Flask(__name__)

def get_predictor():
    # Allow multiple threads to each have their own instance of predictor
    if not hasattr(g, 'predictor'):
        g.predictor = Predictor(service_id)
        g.predictor.start()
    
    return g.predictor

@app.route('/')
def index():
    return 'Predictor is up.'

@app.route('/predict', methods=['POST'])
def predict():
    predictor = get_predictor()
    params = request.get_json()
    query = params['query']
    
    #TODO: check input type
    result = predictor.predict(query)
    return jsonify(result)
