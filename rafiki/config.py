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

# Global
APP_SECRET = os.environ.get('APP_SECRET', 'rafiki')
SUPERADMIN_EMAIL = 'superadmin@rafiki'
SUPERADMIN_PASSWORD = os.environ.get('SUPERADMIN_PASSWORD', 'rafiki')

# Admin
SERVICE_STATUS_WAIT = 1
INFERENCE_WORKER_REPLICAS_PER_TRIAL = 2
INFERENCE_MAX_BEST_TRIALS = 2

# Predictor
PREDICTOR_PREDICT_SLEEP = 0.25

# Inference worker
INFERENCE_WORKER_SLEEP = 0.25
INFERENCE_WORKER_PREDICT_BATCH_SIZE = 32