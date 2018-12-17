import pprint
import time
import requests
import os

from rafiki.client import Client
from rafiki.constants import TaskType, BudgetType, UserType, ModelDependency
from examples.scripts.quickstart import create_user, create_model, \
    create_train_job, get_predictor_host, wait_until_train_job_has_stopped, \
    make_predictions, RAFIKI_HOST, ADMIN_PORT, ADMIN_WEB_PORT, SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD

def run_pos_tagging(client, app, enable_gpu, train_dataset_uri, test_dataset_uri):
    task = TaskType.POS_TAGGING

    print('Adding models to Rafiki...') 
    create_model(client, 'BigramHmm', task, 'examples/models/pos_tagging/BigramHmm.py', \
                'BigramHmm', dependencies={})
    create_model(client, 'PyBiLstm', task, 'examples/models/pos_tagging/PyBiLstm.py', \
                'PyBiLstm', dependencies={ ModelDependency.PYTORCH: '0.4.1' })

    print('Creating train job for app "{}" on Rafiki...'.format(app)) 
    models = [
        {
            'name': 'BigramHmm',
            'budget': {
                BudgetType.MODEL_TRIAL_COUNT: 2,
                BudgetType.ENABLE_GPU: enable_gpu
            }
        },
        {
            'name': 'PyBiLstm',
            'budget': {
                BudgetType.MODEL_TRIAL_COUNT: 2,
                BudgetType.ENABLE_GPU: enable_gpu
            }
        }
    ]
    train_job = client.create_train_job(app, task, train_dataset_uri, test_dataset_uri, models)
    pprint.pprint(train_job)

    print('Waiting for train job to complete...')
    print('This might take a few minutes')
    wait_until_train_job_has_stopped(client, app)
    if not result: raise Exception('Train job has stopped')
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
    rafiki_host = os.environ.get('RAFIKI_IP_ADDRESS', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    user_email = os.environ.get('USER_EMAIL', SUPERADMIN_EMAIL)
    user_password = os.environ.get('USER_PASSWORD', SUPERADMIN_PASSWORD)

    # Initialize client
    client = Client(admin_host=rafiki_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)
    
    app = 'ptb_pos_app'
    enable_gpu = int(os.environ.get('ENABLE_GPU', 0))
    train_dataset_uri = 'https://github.com/nginyc/rafiki-datasets/blob/master/pos_tagging/ptb_for_pos_tagging_train.zip?raw=true'
    test_dataset_uri = 'https://github.com/nginyc/rafiki-datasets/blob/master/pos_tagging/ptb_for_pos_tagging_test.zip?raw=true'

    # Run training & inference
    run_pos_tagging(client, app, enable_gpu, train_dataset_uri, test_dataset_uri)
            