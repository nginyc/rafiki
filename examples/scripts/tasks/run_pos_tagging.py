from pprint import pprint
import time
import requests
import os

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.constants import TaskType, BudgetType, UserType, ModelDependency, ModelAccessRight
from examples.scripts.quickstart import get_predictor_host, \
    wait_until_train_job_has_stopped, make_predictions,  gen_id

from examples.datasets.pos_tagging.load_ptb_format import load_sample_ptb

def run_pos_tagging(client, train_dataset_path, val_dataset_path, enable_gpu):
    task = TaskType.POS_TAGGING

    # Randomly generate app & model names to avoid naming conflicts
    app_id = gen_id()
    app = 'pos_tagging_app_{}'.format(app_id)
    bihmm_model_name = 'BigramHmm_{}'.format(app_id)
    py_model_name = 'PyBiLstm_{}'.format(app_id)

    print('Adding models "{}" and "{}" to Rafiki...'.format(bihmm_model_name, py_model_name)) 
    bihmm_model = client.create_model(bihmm_model_name, task, 'examples/models/pos_tagging/BigramHmm.py', \
                        'BigramHmm', dependencies={}) 
    pprint(bihmm_model)
    py_model = client.create_model(py_model_name, task, 'examples/models/pos_tagging/PyBiLstm.py', \
                        'PyBiLstm', dependencies={ ModelDependency.PYTORCH: '0.4.1' })
    pprint(py_model)
    model_ids = [bihmm_model['id'], py_model['id']]

    print('Creating & uploading datasets onto Rafiki...')
    train_dataset = client.create_dataset('{}_train'.format(app), task, train_dataset_path)
    pprint(train_dataset)
    val_dataset = client.create_dataset('{}_val'.format(app), task, val_dataset_path)
    pprint(val_dataset)

    print('Creating train job for app "{}" on Rafiki...'.format(app))
    budget = {
        BudgetType.MODEL_TRIAL_COUNT: 2,
        BudgetType.ENABLE_GPU: enable_gpu
    }
    train_job = client.create_train_job(app, task, train_dataset['id'], val_dataset['id'], 
                                        budget, models=model_ids)
    pprint(train_job)

    print('Waiting for train job to complete...')
    print('This might take a few minutes')
    wait_until_train_job_has_stopped(client, app)
    print('Train job has been stopped')

    print('Listing best trials of latest train job for app "{}"...'.format(app))
    pprint(client.get_best_trials_of_train_job(app))

    print('Creating inference job for app "{}" on Rafiki...'.format(app))
    pprint(client.create_inference_job(app))
    predictor_host = get_predictor_host(client, app)
    if not predictor_host: raise Exception('Inference job has errored or stopped')
    print('Inference job is running!')

    print('Making predictions for queries:')
    queries = [
        ['Ms.', 'Haag', 'plays', 'Elianti', '18', '.'],
        ['The', 'luxury', 'auto', 'maker', 'last', 'year', 'sold', '1,214', 'cars', 'in', 'the', 'U.S.']
    ]
    print(queries)
    predictions = make_predictions(client, predictor_host, queries)
    print('Predictions are:')
    print(predictions)

    print('Stopping inference job...')
    pprint(client.stop_inference_job(app))

if __name__ == '__main__':
    rafiki_host = os.environ.get('RAFIKI_HOST', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    user_email = os.environ.get('USER_EMAIL', SUPERADMIN_EMAIL)
    user_password = os.environ.get('USER_PASSWORD', SUPERADMIN_PASSWORD)
    out_train_dataset_path = 'data/ptb_for_pos_tagging_train.zip'
    out_val_dataset_path = 'data/ptb_for_pos_tagging_val.zip'

    # Load dataset
    load_sample_ptb(out_train_dataset_path, out_val_dataset_path)

    # Initialize client
    client = Client(admin_host=rafiki_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)
    enable_gpu = int(os.environ.get('ENABLE_GPU', 0))

    # Run training & inference
    run_pos_tagging(client, out_train_dataset_path, out_val_dataset_path, enable_gpu)
            