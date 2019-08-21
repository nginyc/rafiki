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
import time
import pytest

from rafiki.client import Client
from rafiki.constants import TrainJobStatus, BudgetOption, InferenceBudget, InferenceJobStatus
from test.utils import global_setup, make_model_dev, make_app_dev, make_model, make_private_model, \
                    make_invalid_model, make_dataset, gen, superadmin, \
                        wait_for_inference_job_status, make_admin
DATASET_TRAIN_NAME = 'fashion_mnist_train'
# DATASET_TEST_NAME = 'fashion_mnist_test'
DATASET_EVAL_NAME = 'fashion_mnist_eval'
DATASET_TRAIN_FILE_PATH = 'test/workflowdata/fashion_mnist_for_image_classification_train.zip'
# DATASET_TEST_FILE_PATH = 'test/workflowdata/fashion_mnist_test.zip'
DATASET_EVAL_FILE_PATH = 'test/workflowdata/fashion_mnist_for_image_classification_eval.zip'
TASK = 'IMAGE_CLASSIFICATION'

MODEL_NAME = 'TfFeedForward_for_test'
MODEL_CLASS = 'TfFeedForward'
MODEL_PATH = 'test/workflowdata/TfFeedForward.py'
MODEL_DEPENDENCY = {'tensorflow': '1.12.0'}
APP = 'fashion_mnist'

TRAIN_BUDGET = {'MODEL_TRIAL_COUNT': 2, 'GPU_COUNT':0, 'TIME_HOURS':1}
INFERENCE_BUDGET = {'GPU_COUNT': 0}

TRAIN_TIMEOUT = 3600
class TestWorkflow(object):
    train_dataset_info = None
    eval_dataset_info = None
    @pytest.fixture(scope='class')
    def app_dev(self):
        app_dev = make_admin()
        return app_dev

    def test_workflow(self, app_dev):
        assert os.path.exists(DATASET_TRAIN_FILE_PATH)
        assert os.path.exists(DATASET_EVAL_FILE_PATH)
        # app_dev = Client()
        app_dev.get_models()
        train_dataset_info = app_dev.create_dataset(DATASET_TRAIN_NAME, TASK, \
            DATASET_TRAIN_FILE_PATH)
        
        # test_dataset_info = app_dev.create_dataset(DATASET_TEST_NAME, TASK, DATASET_TEST_FILE_PATH)
        eval_dataset_info = app_dev.create_dataset(DATASET_EVAL_NAME, TASK, \
            DATASET_EVAL_FILE_PATH)
        time.sleep(15)
        datasets = app_dev.get_datasets()
        assert any([(x['id'] == train_dataset_info['id']) for x in datasets])
        assert any([(x['id'] == eval_dataset_info['id']) for x in datasets])

        assert os.path.exists(MODEL_PATH)
        model = app_dev.create_model(MODEL_NAME, TASK, MODEL_PATH, MODEL_CLASS, MODEL_DEPENDENCY)
        assert 'id' in model
        assert model['name'] == MODEL_NAME

        # Available models should contain created model
        models = app_dev.get_available_models()
        assert any([x['id'] == model['id'] for x in models])

        train_job_create = app_dev.create_train_job(APP, TASK, train_dataset_info['id'], \
            eval_dataset_info['id'], TRAIN_BUDGET, [model['id']])
        assert 'id' in train_job_create
        wait_for_train_job_status(app_dev, APP, TrainJobStatus.STOPPED, TRAIN_TIMEOUT)
        train_job = app_dev.get_train_job(APP)
        assert train_job['id'] == train_job_create['id']
        assert train_job['app'] == train_job_create['app']
        assert train_job['status'] == TrainJobStatus.STOPPED

        inference_job_create = app_dev.create_inference_job(APP, -1, INFERENCE_BUDGET)
        assert 'id' in inference_job_create
        wait_for_inference_job_status(app_dev, APP, InferenceJobStatus.RUNNING)
        inference_job = app_dev.get_running_inference_job(APP)
        assert inference_job['id'] == inference_job_create['id']
        assert inference_job['app'] == inference_job_create['app']
        assert inference_job['status'] == InferenceJobStatus.RUNNING
        assert all([(x in inference_job) for x in ['predictor_host']])

        inference_job = app_dev.stop_inference_job(APP)
        assert inference_job['id'] == inference_job_create['id']
        assert inference_job['app'] == APP

        # Will throw error when inference job is not longer running
        with pytest.raises(Exception, match='InvalidRunningInferenceJobError'):
            wait_for_inference_job_status(app_dev, APP, InferenceJobStatus.STOPPED)

# Raise exception if train job errors or it timeouts
def wait_for_train_job_status(client: Client, app, status, timeout):
    length = 0
    tick = 3

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