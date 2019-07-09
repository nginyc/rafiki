from pprint import pprint
import time
import requests
import argparse
import traceback
import os
import string
import random

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import TaskType, UserType, BudgetType, TrainJobStatus, \
                                InferenceJobStatus, ModelDependency, ModelAccessRight

from examples.datasets.image_files.load_fashion_mnist import load_fashion_mnist

# Generates a random ID
def gen_id(length=16):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def wait_until_train_job_has_stopped(client, app, timeout=60*20, tick=10):
    length = 0
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

# Returns `predictor_host` of inference job
def get_predictor_host(client, app):
    while True:
        try:
            inference_job = client.get_running_inference_job(app)
            status = inference_job.get('status')
            if status == InferenceJobStatus.RUNNING:
                return inference_job.get('predictor_host')
            elif status in [InferenceJobStatus.ERRORED, InferenceJobStatus.STOPPED]:
                # Inference job has either errored or been stopped
                return False
            else:
                time.sleep(10)
                continue
        except:
            pass

def make_predictions(client, predictor_host, queries):
    predictions = []

    for query in queries:
        res = requests.post(
            url='http://{}/predict'.format(predictor_host),
            json={ 'query': query }
        )

        if res.status_code != 200:
            raise Exception(res.text)

        predictions.append(res.json()['prediction'])

    return predictions


def quickstart(client, train_dataset_path, val_dataset_path, gpus, trials):
    '''
    Runs a sample full train-inference flow for the task ``IMAGE_CLASSIFICATION``.
    '''

    task = TaskType.IMAGE_CLASSIFICATION

    # Randomly generate app & model names to avoid naming conflicts
    app_id = gen_id()
    app = 'image_classification_app_{}'.format(app_id)
    tf_model_name = 'TfFeedForward_{}'.format(app_id)
    sk_model_name = 'SkDt_{}'.format(app_id)

    print('Preprocessing datasets...')
    load_fashion_mnist(train_dataset_path, val_dataset_path)
    
    print('Creating & uploading datasets onto Rafiki...')
    train_dataset = client.create_dataset('{}_train'.format(app), task, train_dataset_path)
    pprint(train_dataset)
    val_dataset = client.create_dataset('{}_val'.format(app), task, val_dataset_path)
    pprint(val_dataset)

    print('Adding models "{}" and "{}" to Rafiki...'.format(tf_model_name, sk_model_name)) 
    tf_model = client.create_model(tf_model_name, task, 'examples/models/image_classification/TfFeedForward.py', 
                        'TfFeedForward', dependencies={ ModelDependency.TENSORFLOW: '1.12.0' })
    pprint(tf_model)
    sk_model = client.create_model(sk_model_name, task, 'examples/models/image_classification/SkDt.py', 
                        'SkDt', dependencies={ ModelDependency.SCIKIT_LEARN: '0.20.0' })
    pprint(sk_model)
    model_ids = [tf_model['id'], sk_model['id']]

    print('Creating train job for app "{}" on Rafiki...'.format(app)) 
    budget = {
        BudgetType.MODEL_TRIAL_COUNT: trials,
        BudgetType.GPU_COUNT: gpus
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
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], 
        [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], 
        [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], 
        [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], 
        [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], 
        [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], 
        [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], 
        [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], 
        [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ]
    print(queries)
    predictions = make_predictions(client, predictor_host, queries)
    print('Predictions are:')
    print(predictions)

    print('Stopping inference job...')
    pprint(client.stop_inference_job(app))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='Host of Rafiki instance')
    parser.add_argument('--web_admin_port', type=int, default=os.environ.get('WEB_ADMIN_EXT_PORT', 3001), help='Port for Rafiki Admin Web on host')
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use')
    parser.add_argument('--trials', type=int, default=5, help='How many trials to conduct for each model')
    (args, _) = parser.parse_known_args()
    out_train_dataset_path = 'data/fashion_mnist_for_image_classification_train.zip'
    out_val_dataset_path = 'data/fashion_mnist_for_image_classification_val.zip'

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)
    web_admin_url = 'http://{}:{}'.format(args.host, args.web_admin_port)
    print('During training, you can view the status of the train job at {}'.format(web_admin_url))
    print('Login with email "{}" and password "{}"'.format(args.email, args.password)) 
    
    quickstart(client, out_train_dataset_path, out_val_dataset_path, args.gpus, args.trials)
