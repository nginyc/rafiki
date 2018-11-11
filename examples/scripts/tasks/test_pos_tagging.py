import pprint
import time
import requests

from rafiki.client import Client
from rafiki.constants import TaskType, BudgetType, UserType
from examples.scripts.client_quickstart import create_user, create_model, \
    create_train_job, wait_until_inference_job_is_running, wait_until_train_job_has_completed, \
    make_predictions, RAFIKI_HOST, ADMIN_PORT, ADMIN_WEB_PORT, SUPERADMIN_EMAIL, MODEL_DEVELOPER_EMAIL, \
    APP_DEVELOPER_EMAIL, USER_PASSWORD

if __name__ == '__main__':
    app = 'ptb_pos_app'
    task = TaskType.POS_TAGGING
    train_dataset_uri = 'data/corpus.train.zip'
    test_dataset_uri = 'data/corpus.test.zip'
    queries = [
        ['Ms.', 'Haag', 'plays', 'Elianti', '18', '.'],
        ['The', 'luxury', 'auto', 'maker', 'last', 'year', 'sold', '1,214', 'cars', 'in', 'the', 'U.S.']
    ]

    client = Client(admin_host=RAFIKI_HOST, admin_port=ADMIN_PORT)
    client.login(email=SUPERADMIN_EMAIL, password=USER_PASSWORD)

    print('Creating model developer in Rafiki...')
    create_user(client, MODEL_DEVELOPER_EMAIL, USER_PASSWORD, UserType.MODEL_DEVELOPER)

    print('Creating app developer in Rafiki...')
    create_user(client, APP_DEVELOPER_EMAIL, USER_PASSWORD, UserType.APP_DEVELOPER)

    print('Logging in as model developer...')
    client.login(email=MODEL_DEVELOPER_EMAIL, password=USER_PASSWORD)

    print('Adding models to Rafiki...') 
    create_model(client, 'BigramHmm', task, \
                'examples/models/pos_tagging/BigramHmm.py', 'BigramHmm')

    print('Logging in as app developer...')
    client.login(email=APP_DEVELOPER_EMAIL, password=USER_PASSWORD)

    print('Creating train job for app "{}" on Rafiki...'.format(app)) 
    (train_job, train_job_web_url) = create_train_job(client, app, task, 
                                            train_dataset_uri, test_dataset_uri)
    pprint.pprint(train_job)

    print('Waiting for train job to complete...')
    result = wait_until_train_job_has_completed(client, app)
    if not result: raise Exception('Train job has errored or stopped')
    print('Train job has been completed!')

    print('Listing best trials of latest train job for app "{}"...'.format(app))
    pprint.pprint(client.get_best_trials_of_train_job(app))

    print('Creating inference job for app "{}" on Rafiki...'.format(app))
    pprint.pprint(client.create_inference_job(app))

    print('Waiting for inference job to be running...')
    predictor_host = wait_until_inference_job_is_running(client, app)
    if not predictor_host: raise Exception('Inference job has errored or stopped')
    print('Inference job is running!')

    print('Making predictions for queries:')
    print(queries)
    predictions = make_predictions(client, predictor_host, queries)
    print('Predictions are:')
    print(predictions)

    print('Stopping inference job...')
    pprint.pprint(client.stop_inference_job(app))


