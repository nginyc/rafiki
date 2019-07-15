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
from rafiki.constants import TrainJobStatus, BudgetOption
from test.utils import global_setup, make_model_dev, make_app_dev, make_model, make_private_model, \
                    make_invalid_model, make_dataset, gen, superadmin, wait_for_train_job_status

class TestTrainJobs():

    @pytest.fixture(scope='class')
    def app_dev_create_train_job_and_waited(self):
        app_dev = make_app_dev()
        (task, app, model_id, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev)

        # Create train job
        train_job = app_dev.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])
        assert 'id' in train_job
        wait_for_train_job_status(app_dev, app, TrainJobStatus.STOPPED)

        return (app_dev, app, train_job['id'], task)

    def test_app_dev_create_train_job(self, app_dev_create_train_job_and_waited):
        (app_dev, app, train_job_id, *args) = app_dev_create_train_job_and_waited
        app_dev: Client

        # View train job
        train_job = app_dev.get_train_job(app)
        assert train_job['id'] == train_job_id
        assert train_job['app'] == app
        assert train_job['status'] == TrainJobStatus.STOPPED
        
        # Get train job by user
        user = app_dev.get_current_user()
        user_id = user['id']
        train_jobs = app_dev.get_train_jobs_by_user(user_id)
        assert any([x['id'] == train_job_id for x in train_jobs])

        # Get train job by app
        train_jobs = app_dev.get_train_jobs_of_app(app)
        assert any([x['id'] == train_job_id for x in train_jobs])

    def test_app_dev_get_trials(self, app_dev_create_train_job_and_waited):
        (app_dev, app, *args) = app_dev_create_train_job_and_waited
        app_dev: Client

        # Get trials of stopped train job
        trials = app_dev.get_trials_of_train_job(app)
        assert len(trials) > 0

        # Get best trials of stopped train job
        best_trials = app_dev.get_best_trials_of_train_job(app)
        assert len(best_trials) > 0


    def test_app_dev_get_trial(self, app_dev_create_train_job_and_waited):
        (app_dev, app, *args) = app_dev_create_train_job_and_waited
        app_dev: Client

        # Get a trial
        trials = app_dev.get_trials_of_train_job(app)
        assert len(trials) > 0
        trial = trials[0]
        assert 'id' in trial
        trial_id = trial['id']

        # Get info for a trial
        trial = app_dev.get_trial(trial_id)
        assert trial['id'] == trial_id
        assert all([(x in trial) for x in ['proposal', 'status', 'score', 'datetime_started', 'datetime_stopped']])
        
    
    def test_app_dev_get_trial_logs(self, app_dev_create_train_job_and_waited):
        (app_dev, app, *args) = app_dev_create_train_job_and_waited
        app_dev: Client

        # Get a trial
        trials = app_dev.get_trials_of_train_job(app)
        assert len(trials) > 0
        trial = trials[0]
        assert 'id' in trial
        trial_id = trial['id']

        # Get logs for trial
        logs = app_dev.get_trial_logs(trial_id)
        assert len(logs) > 0


    def test_app_dev_create_2nd_app_version(self, app_dev_create_train_job_and_waited):
        (app_dev, app, task, *args) = app_dev_create_train_job_and_waited
        (_, _, model_id, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev, task=task) # Get another set of job info
        app_dev: Client
        
        # Create another train job
        train_job = app_dev.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])
        assert train_job['app'] == app
        assert train_job['app_version'] == 2 # 2nd version of the train job


    def test_multiple_app_devs_use_same_app(self, app_dev_create_train_job_and_waited):
        (app_dev, app, task, *args) = app_dev_create_train_job_and_waited
        app_dev2 = make_app_dev()
        (_, _, model_id, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev2, task=task) # Get another set of job info
        
        # App dev 2 create another train job with same app
        train_job = app_dev2.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])
        assert train_job['app'] == app
        assert train_job['app_version'] == 1 # Should not increment


    def test_app_dev_cant_view_others_job(self, app_dev_create_train_job_and_waited):
        (app_dev, app, task, *args) = app_dev_create_train_job_and_waited
        app_dev: Client
        app_dev_user = app_dev.get_current_user()
        app_dev_id = app_dev_user['id']
        app_dev2 = make_app_dev()

        with pytest.raises(Exception):
            app_dev2.get_train_jobs_by_user(app_dev_id)


    def test_app_dev_stop_train_job(self):
        app_dev = make_app_dev()
        (task, app, model_id, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev)

        # Create train job
        train_job = app_dev.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])
        assert 'id' in train_job

        # Stop train job
        app_dev.stop_train_job(app)

        # Train job should have stopped
        train_job = app_dev.get_train_job(app)
        assert train_job['status'] == TrainJobStatus.STOPPED
    

    def test_app_dev_create_train_job_with_gpu(self):
        app_dev = make_app_dev()
        (task, app, model_id, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev)
        budget[BudgetOption.GPU_COUNT] = 1 # With GPU
        
        # Create train job
        train_job = app_dev.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])
        assert 'id' in train_job

        # Wait until train job stops
        wait_for_train_job_status(app_dev, app, TrainJobStatus.STOPPED)

        # Train job should have stopped without error
        train_job = app_dev.get_train_job(app)
        assert train_job['status'] == TrainJobStatus.STOPPED

        # Train job should have trials
        trials = app_dev.get_trials_of_train_job(app)
        assert len(trials) > 0

    def test_app_dev_cant_use_private_model(self):
        app_dev = make_app_dev()
        (task, app, _, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev)
        model_id = make_private_model(task=task) # Have private model created

        # Can't create train job with private model
        with pytest.raises(Exception, match='InvalidModelError'):
            app_dev.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])

    def test_app_dev_informed_model_error(self):
        app_dev = make_app_dev()
        (task, app, _, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev)
        model_id = make_invalid_model(task=task) # Have invalid model created

        # Create train job
        app_dev.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id])
        
        # Train job will be errored
        with pytest.raises(Exception, match='errored'):
            wait_for_train_job_status(app_dev, app, TrainJobStatus.STOPPED)

    def test_app_dev_informed_model_error_with_multiple_models(self):
        app_dev = make_app_dev()
        (task, app, _, train_dataset_id, val_dataset_id, budget) = make_train_job_info(app_dev)
        model_id = make_invalid_model(task=task) # Have invalid model created
        model_id2 = make_model(task=task) # Have valid model created

        # Create train job
        app_dev.create_train_job(app, task, train_dataset_id, val_dataset_id, budget, models=[model_id, model_id2])

        # Train job will be errored
        with pytest.raises(Exception, match='errored'):
            wait_for_train_job_status(app_dev, app, TrainJobStatus.STOPPED)

def make_train_job_info(client: Client, task=None):
    task = task or gen()
    app = gen()
    train_dataset_id = make_dataset(client, task=task)
    val_dataset_id = make_dataset(client, task=task)
    model_id = make_model(task=task)

    # 1 trial & no GPU
    budget = {BudgetOption.MODEL_TRIAL_COUNT: 1, BudgetOption.GPU_COUNT: 0}

    return (task, app, model_id, train_dataset_id, val_dataset_id, budget)