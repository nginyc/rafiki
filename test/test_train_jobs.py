import pytest
import tempfile
import os
import time

from rafiki.client import Client
from rafiki.constants import ModelAccessRight, TrainJobStatus
from test.utils import make_model_dev, make_app_dev, make_model, make_private_model, \
                        gen, superadmin, DATASET_TRAIN_FILE_PATH, DATASET_VAL_FILE_PATH

TRAIN_JOB_TIMEOUT_SECS = 5 * 60
    
class TestTrainJobs():

    @pytest.fixture(scope='class')
    def app_dev_create_train_job_and_waited(self):
        (task, app, model_id, train_dataset_uri, val_dataset_uri, budget) = make_train_job_info()
        app_dev = make_app_dev()

        # Create train job
        train_job = app_dev.create_train_job(app, task, train_dataset_uri, val_dataset_uri, budget, models=[model_id])
        assert 'id' in train_job
        train_job_id = train_job['id']

        # Wait until train job stops
        wait_until_train_job_stops(app, app_dev)

        return (app_dev, app, train_job_id, task)

    def test_app_dev_create_train_job(self, app_dev_create_train_job_and_waited):
        (app_dev, app, train_job_id, *args) = app_dev_create_train_job_and_waited
        app_dev: Client

        # View train job
        train_job = app_dev.get_train_job(app)
        assert train_job['id'] == train_job_id
        assert train_job['app'] == app
        assert 'status' in train_job
        
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
        assert all([(x in trial) for x in ['knobs', 'status', 'score', 'datetime_started', 'datetime_stopped']])
        
    
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
        (_, _, model_id, train_dataset_uri, val_dataset_uri, budget) = make_train_job_info(task=task) # Get another set of job info
        app_dev: Client
        
        # Create another train job
        train_job = app_dev.create_train_job(app, task, train_dataset_uri, val_dataset_uri, budget, models=[model_id])
        assert train_job['app'] == app
        assert train_job['app_version'] == 2 # 2nd version of the train job


    def test_app_dev_stop_train_job(self):
        (task, app, model_id, train_dataset_uri, val_dataset_uri, budget) = make_train_job_info()
        app_dev = make_app_dev()

        # Create train job
        train_job = app_dev.create_train_job(app, task, train_dataset_uri, val_dataset_uri, budget, models=[model_id])
        assert 'id' in train_job

        # Stop train job
        app_dev.stop_train_job(app)

        # Train job should have stopped
        train_job = app_dev.get_train_job(app)
        assert train_job['status'] == TrainJobStatus.STOPPED
    

    def test_app_dev_create_train_job_with_gpu(self):
        (task, app, model_id, train_dataset_uri, val_dataset_uri, budget) = make_train_job_info()
        budget['GPU_COUNT'] = 1 # Activate GPU
        app_dev = make_app_dev()
        
        # Create train job
        train_job = app_dev.create_train_job(app, task, train_dataset_uri, val_dataset_uri, budget, models=[model_id])
        assert 'id' in train_job

        # Wait until train job stops
        wait_until_train_job_stops(app, app_dev)

        # Train job should have stopped without error
        train_job = app_dev.get_train_job(app)
        assert train_job['status'] == TrainJobStatus.STOPPED

        # Train job should have trials
        trials = app_dev.get_trials_of_train_job(app)
        assert len(trials) > 0


    def test_app_dev_cant_use_private_model(self):
        (task, app, model_id, train_dataset_uri, val_dataset_uri, budget) = make_train_job_info()
        model_id = make_private_model() # Have private model created
        app_dev = make_app_dev()

        # Can't create train job with private model
        with pytest.raises(Exception):
            app_dev.create_train_job(app, task, train_dataset_uri, val_dataset_uri, budget, models=[model_id])


def make_train_job_info(task=None):
    task = task or gen()
    app = gen()
    model_id = make_model(task=task)
    train_dataset_uri = DATASET_TRAIN_FILE_PATH
    val_dataset_uri = DATASET_VAL_FILE_PATH
    budget = { 'MODEL_TRIAL_COUNT': 1 }

    return (task, app, model_id, train_dataset_uri, val_dataset_uri, budget)


def wait_until_train_job_stops(app, client: Client):
    length = 0
    timeout = TRAIN_JOB_TIMEOUT_SECS
    tick = 1

    while True:
        train_job = client.get_train_job(app)
        status = train_job['status']
        if status not in [TrainJobStatus.STARTED, TrainJobStatus.RUNNING]:
            # Train job has stopped
            return
            
        # Still running...
        if length >= timeout:
            raise TimeoutError('Train job is running for too long')

        length += tick
        time.sleep(tick)
