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

from rafiki.client import Client
from rafiki.constants import TrainJobStatus, InferenceJobStatus
from test.utils import global_setup, make_app_dev, make_model, gen,make_train_job, \
                    wait_for_train_job_status, wait_for_inference_job_status

MODEL_INVALID_LOAD_PARAMS_PATH = 'test/data/ModelInvalidLoadParams.py'

class TestInferenceJobs():

    @pytest.fixture(scope='class')
    def app_dev_create_inference_job_and_waited(self):
        app_dev = make_app_dev()
        (task, app, train_job_id) = make_inference_job_info(app_dev)

        # Create inference job
        inference_job = app_dev.create_inference_job(app)
        assert 'id' in inference_job
        wait_for_inference_job_status(app_dev, app, InferenceJobStatus.RUNNING)

        return (app_dev, app, inference_job['id'], task)

    def test_app_dev_create_inference_job(self, app_dev_create_inference_job_and_waited):
        (app_dev, app, inference_job_id, *args) = app_dev_create_inference_job_and_waited
        app_dev: Client

        # View inference job
        job = app_dev.get_running_inference_job(app)
        assert job['id'] == inference_job_id
        assert job['app'] == app
        assert job['status'] == InferenceJobStatus.RUNNING
        assert all([(x in job) for x in ['predictor_host']])

        # Get inference job by user
        user = app_dev.get_current_user()
        user_id = user['id']
        inference_job = app_dev.get_inference_jobs_by_user(user_id)
        assert any([x['id'] == inference_job_id for x in inference_job])

        # Get inference job by app
        inference_jobs = app_dev.get_inference_jobs_of_app(app)
        assert any([x['id'] == inference_job_id for x in inference_jobs])

    def test_app_dev_stop_inference_job(self, app_dev_create_inference_job_and_waited):
        (app_dev, app, inference_job_id, *args) = app_dev_create_inference_job_and_waited
        app_dev: Client

        # Stop inference job
        job = app_dev.stop_inference_job(app)
        assert job['id'] == inference_job_id
        assert job['app'] == app

        # Will throw error when inference job is not longer running
        with pytest.raises(Exception, match='InvalidRunningInferenceJobError'):
            wait_for_inference_job_status(app_dev, app, InferenceJobStatus.STOPPED)

    def test_app_dev_informed_model_error_in_inference(self):
        app_dev = make_app_dev()
        task = gen()
        model_id = make_model(task=task, model_file_path=MODEL_INVALID_LOAD_PARAMS_PATH)
        (task, app, train_job_id) = make_inference_job_info(app_dev, task=task, model_id=model_id)
        
        # Create inference job
        inference_job = app_dev.create_inference_job(app)

        # Will eventuallly throw error as inference job is no longer running
        with pytest.raises(Exception, match='InvalidRunningInferenceJobError'):
            wait_for_inference_job_status(app_dev, app, InferenceJobStatus.ERRORED)

        # Inference job errored
        inference_jobs = app_dev.get_inference_jobs_of_app(app)
        inference_job = next(x for x in inference_jobs if x['id'] == inference_job['id'])
        assert inference_job['status'] == InferenceJobStatus.ERRORED

    def test_app_dev_cant_use_running_train_job(self):
        app_dev = make_app_dev()
        app = gen()
        train_job_id = make_train_job(app_dev, app=app)

        # Will throw error as train job is still running
        with pytest.raises(Exception, match='InvalidTrainJobError'):
            app_dev.create_inference_job(app)

def make_inference_job_info(client: Client, task=None, model_id=None):
    task = task or gen()
    app = gen()
    train_job_id = make_train_job(client, task=task, app=app, model_id=model_id)
    wait_for_train_job_status(client, app, TrainJobStatus.STOPPED)
    return (task, app, train_job_id)
