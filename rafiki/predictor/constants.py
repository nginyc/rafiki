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

from typing import Union, Any
import uuid

class Query():
    def __init__(self, query: Any):
        self.id = str(uuid.uuid4())
        self.query = query
    
    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.id == other.id 
                and self.query == other.query)

class Prediction():
    def __init__(self, 
                # Raw prediction, or None if the worker is unable to make a prediction (e.g. errored)
                prediction: Union[Any, None], 
                # ID of query of prediction
                query_id: str, 
                # Worker who made the prediction, if any
                worker_id: str = None): 
        self.prediction = prediction
        self.query_id = query_id
        self.worker_id = worker_id

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.prediction == other.prediction 
                and self.query_id == other.query_id
                and self.worker_id == other.worker_id)