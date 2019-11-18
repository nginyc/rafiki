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

from typing import Dict, Any

class BudgetOption():
    GPU_COUNT = 'GPU_COUNT'
    TIME_HOURS = 'TIME_HOURS'
    MODEL_TRIAL_COUNT = 'MODEL_TRIAL_COUNT'

Budget = Dict[BudgetOption, Any]

class InferenceBudgetOption():
    GPU_COUNT = 'GPU_COUNT'

InferenceBudget = Dict[InferenceBudgetOption, Any]

ModelDependencies = Dict[str, str]

class ModelAccessRight():
    PUBLIC = 'PUBLIC'
    PRIVATE = 'PRIVATE'

class InferenceJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class TrainJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERRORED = 'ERRORED'

class TrialStatus():
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    COMPLETED = 'COMPLETED'

class UserType():
    SUPERADMIN = 'SUPERADMIN'
    ADMIN = 'ADMIN'
    MODEL_DEVELOPER = 'MODEL_DEVELOPER'
    APP_DEVELOPER = 'APP_DEVELOPER'

class ServiceType():
    TRAIN = 'TRAIN'
    ADVISOR = 'ADVISOR'
    PREDICT = 'PREDICT'
    INFERENCE = 'INFERENCE'

class ServiceStatus():
    STARTED = 'STARTED'
    DEPLOYING = 'DEPLOYING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class ModelDependency():
    TENSORFLOW = 'tensorflow'
    KERAS = 'Keras'
    SCIKIT_LEARN = 'scikit-learn'
    TORCH = 'torch'
    TORCHVISION = 'torchvision'
    SINGA = 'singa'
    XGBOOST = 'xgboost'
    DS_CTCDECODER = 'ds-ctcdecoder'
    NLTK = 'nltk'
    SKLEARN_CRFSUITE = 'sklearn-crfsuite'

