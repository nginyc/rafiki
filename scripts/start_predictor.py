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

from rafiki.utils.service import run_service
from rafiki.db import Database
from rafiki.predictor.app import app

def start_service(service_id, service_type):
    app.run(host='0.0.0.0', 
            port=os.getenv('PREDICTOR_PORT', 3003), 
            threaded=True)

def end_service(service_id, service_type):
    pass

db = Database()
run_service(db, start_service, end_service)
