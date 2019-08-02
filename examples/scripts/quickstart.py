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

from pprint import pprint
import time
import requests
import argparse
import os

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import BudgetOption, InferenceBudgetOption, InferenceJobStatus, ModelDependency
from rafiki.model import utils

from examples.scripts.utils import gen_id, wait_until_train_job_has_stopped
from examples.datasets.image_files.load_fashion_mnist import load_fashion_mnist

# Returns `predictor_host` of inference job
def get_predictor_host(client, app):
    while True:
            inference_job = client.get_running_inference_job(app)
            status = inference_job.get('status')
            if status == InferenceJobStatus.RUNNING:
                return inference_job.get('predictor_host')
            else:
                time.sleep(10)

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


def quickstart(client, train_dataset_path, val_dataset_path, gpus, hours, query_paths):
    '''
        Conducts a full train-inference flow on the Fashion MNIST dataset with
        models `SkDt` and `TfFeedForward` for the task `IMAGE_CLASSIFICATION`.
    '''

    task = 'IMAGE_CLASSIFICATION'

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
        BudgetOption.TIME_HOURS: hours,
        BudgetOption.GPU_COUNT: gpus
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
    if not predictor_host: raise Exception('Inference job has errored')
    print('Inference job is running!')

    print('Making predictions for query images:')
    print(query_paths)
    queries = utils.dataset.load_images(query_paths).tolist()
    predictions = make_predictions(client, predictor_host, queries)
    print('Predictions are:')
    print(predictions)

    print('Stopping inference job...')
    pprint(client.stop_inference_job(app))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='Host of Rafiki instance')
    parser.add_argument('--web_admin_port', type=int, default=os.environ.get('WEB_ADMIN_EXT_PORT', 3001), help='Port for Rafiki Web Admin on host')
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use for training')
    parser.add_argument('--hours', type=float, default=0.1, help='How long the train job should run for (in hours)') # 6min
    parser.add_argument('--query_path', type=str, 
                        default='examples/data/image_classification/fashion_mnist_test_1.png,examples/data/image_classification/fashion_mnist_test_2.png', 
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()
    out_train_dataset_path = 'data/fashion_mnist_train.zip'
    out_val_dataset_path = 'data/fashion_mnist_val.zip'

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)
    web_admin_url = 'http://{}:{}'.format(args.host, args.web_admin_port)
    print('During training, you can view the status of the train job at {}'.format(web_admin_url))
    print('Login with email "{}" and password "{}"'.format(args.email, args.password)) 
    
    # Run quickstart
    quickstart(client, out_train_dataset_path, out_val_dataset_path, args.gpus, args.hours, args.query_path.split(','))
