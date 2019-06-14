import pprint
import time
import requests
import argparse
import os

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import TaskType, BudgetType, UserType, ModelDependency, ModelAccessRight
from examples.scripts.quickstart import get_predictor_host, \
    wait_until_train_job_has_stopped, make_predictions,  gen_id

def run_pos_tagging(client, gpus):
    task = TaskType.POS_TAGGING

    # Randomly generate app & model names to avoid naming conflicts
    app_id = gen_id()
    app = 'pos_tagging_app_{}'.format(app_id)
    bihmm_model_name = 'BigramHmm_{}'.format(app_id)
    py_model_name = 'PyBiLstm_{}'.format(app_id)

    print('Adding models "{}" and "{}" to Rafiki...'.format(bihmm_model_name, py_model_name)) 
    bihmm_model = client.create_model(bihmm_model_name, task, 'examples/models/pos_tagging/BigramHmm.py', \
                        'BigramHmm', dependencies={}) 
    py_model = client.create_model(py_model_name, task, 'examples/models/pos_tagging/PyBiLstm.py', \
                        'PyBiLstm', dependencies={ ModelDependency.PYTORCH: '0.4.1' })
    model_ids = [bihmm_model['id'], py_model['id']]

    print('Creating train job for app "{}" on Rafiki...'.format(app))
    budget = {
        BudgetType.MODEL_TRIAL_COUNT: 5,
        BudgetType.GPU_COUNT: gpus
    }
    train_dataset_uri = 'https://github.com/nginyc/rafiki-datasets/blob/master/pos_tagging/ptb_for_pos_tagging_train.zip?raw=true'
    test_dataset_uri = 'https://github.com/nginyc/rafiki-datasets/blob/master/pos_tagging/ptb_for_pos_tagging_test.zip?raw=true'
    train_job = client.create_train_job(app, task, train_dataset_uri, test_dataset_uri, 
                                        budget, models=model_ids)
    pprint.pprint(train_job)

    print('Waiting for train job to complete...')
    print('This might take a few minutes')
    wait_until_train_job_has_stopped(client, app)
    print('Train job has been stopped')

    print('Listing best trials of latest train job for app "{}"...'.format(app))
    pprint.pprint(client.get_best_trials_of_train_job(app))

    print('Creating inference job for app "{}" on Rafiki...'.format(app))
    pprint.pprint(client.create_inference_job(app))
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
    pprint.pprint(client.stop_inference_job(app))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)
    
    # Run quickstart
    run_pos_tagging(client, args.gpus)