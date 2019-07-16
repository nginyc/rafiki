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
import os
import time
import uuid
import random
import numpy as np

from rafiki.constants import UserType, ModelAccessRight, TrainJobStatus, BudgetOption, InferenceJobStatus
from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL


superadmin_email = SUPERADMIN_EMAIL
superadmin_password = os.environ['SUPERADMIN_PASSWORD']

# Details for mocks
DATASET_FILE_PATH = 'test/data/dataset.csv'
MODEL_FILE_PATH = 'test/data/Model.py'
MODEL_INVALID_FILE_PATH = 'test/data/ModelInvalid.py'
MODEL_CLASS = 'Model'
JOB_TIMEOUT_SECS = 60

####################################
# General
####################################

@pytest.fixture(scope='session', autouse=True)
def global_setup():
    random.seed(0)
    np.random.seed(0)

def gen():
    return str(uuid.uuid4())

def gen_email():
    return f'{uuid.uuid4()}@rafiki'

####################################
# Users
####################################

@pytest.fixture(scope='module')
def superadmin():
    client = Client()
    client.login(superadmin_email, superadmin_password)
    return client

def make_admin(**kwargs):
    return make_user(UserType.ADMIN, **kwargs)

def make_app_dev(**kwargs):
    return make_user(UserType.APP_DEVELOPER, **kwargs)

def make_model_dev(**kwargs):
    return make_user(UserType.MODEL_DEVELOPER, **kwargs)

# Make a client logged in as new user with a specific user type
def make_user(user_type, email=None, password=None):
    email = email or gen_email()
    password = password or gen()
    client = Client()
    client.login(superadmin_email, superadmin_password)
    client.create_user(email, password, user_type)
    client.login(email, password)
    return client
    

####################################
# Datasets
####################################

def make_dataset(client: Client, task=None):
    name = gen()
    task = task or gen()
    file_path = DATASET_FILE_PATH
    dataset = client.create_dataset(name, task, file_path)
    dataset_id = dataset['id']
    return dataset_id

####################################
# Models
####################################

def make_model(task=None, access_right=ModelAccessRight.PUBLIC, model_file_path=MODEL_FILE_PATH):
    model_dev = make_model_dev()
    task = task or gen()
    name = gen()
    model_class = MODEL_CLASS
    dependencies = {}
    model = model_dev.create_model(name, task, model_file_path, model_class, dependencies, access_right=access_right)
    model_id = model['id']
    return model_id

def make_private_model(**kwargs):
    return make_model(access_right=ModelAccessRight.PRIVATE, **kwargs)

def make_invalid_model(**kwargs):
    return make_model(model_file_path=MODEL_INVALID_FILE_PATH, **kwargs)

####################################
# Train Jobs
####################################

def make_train_job(client: Client, task=None, app=None, model_id=None):
    task = task or gen()
    app = app or gen()
    train_dataset_id = make_dataset(client, task=task)
    val_dataset_id = make_dataset(client, task=task)
    model_id = model_id or make_model(task=task)
    budget = {BudgetOption.MODEL_TRIAL_COUNT: 1, BudgetOption.GPU_COUNT: 0}
    train_job = client.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])
    return train_job['id']

# Raise exception if train job errors or it timeouts
def wait_for_train_job_status(client: Client, app, status):
    length = 0
    timeout = JOB_TIMEOUT_SECS
    tick = 1

    while True:
        train_job = client.get_train_job(app)
        if train_job['status'] == status:
            return
        elif train_job['status'] == TrainJobStatus.ERRORED:
            raise Exception('Train job has errored')
            
        # Still running...
        if length >= timeout:
            raise TimeoutError('Waiting for too long')

        length += tick
        time.sleep(tick)

####################################
# Inference Jobs
####################################

# Raise exception if inference job errors or it timeouts
def wait_for_inference_job_status(client: Client, app, status):
    length = 0
    timeout = JOB_TIMEOUT_SECS
    tick = 1

    while True:
        inference_job = client.get_running_inference_job(app)
        if inference_job['status'] == status:
            return
        elif inference_job['status'] == InferenceJobStatus.ERRORED:
            raise Exception('Inference job has errored')
            
        # Still running...
        if length >= timeout:
            raise TimeoutError('Waiting for too long')

        length += tick
        time.sleep(tick)

