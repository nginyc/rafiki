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
import argparse
import os

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import BudgetOption, ModelDependency

from examples.scripts.quickstart import get_predictor_host, \
    wait_until_train_job_has_stopped, make_predictions, gen_id

from examples.datasets.tabular.csv_file import load

def run_tabular_regression(client, csv_file_url, gpus, hours, features=None, target=None, queries=None):
    '''
    Runs a sample full train-inference flow for the task ``TABULAR_REGRESSION``.
    '''

    task = 'TABULAR_REGRESSION'

    # Randomly generate app & model names to avoid naming conflicts
    app_id = gen_id()
    app = 'tabular_regression_app_{}'.format(app_id)
    xgb_model_name = 'XgbReg_{}'.format(app_id)
    train_dataset_path = 'data/{}_train.csv'.format(app)
    val_dataset_path = 'data/{}_val.csv'.format(app)

    print('Preprocessing dataset...')
    load(csv_file_url, train_dataset_path, val_dataset_path)

    print('Creating & uploading datasets onto Rafiki...')
    train_dataset = client.create_dataset('{}_train'.format(app), task, train_dataset_path)
    pprint(train_dataset)
    val_dataset = client.create_dataset('{}_val'.format(app), task, val_dataset_path)
    pprint(val_dataset)

    print('Adding models "{}" to Rafiki...'.format(xgb_model_name)) 
    xgb_model = client.create_model(xgb_model_name, task, 'examples/models/tabular_regression/XgbReg.py', \
                        'XgbReg', dependencies={ ModelDependency.XGBOOST: '0.90' }) 
    pprint(xgb_model)

    print('Creating train job for app "{}" on Rafiki...'.format(app))
    budget = {
        BudgetOption.TIME_HOURS: hours,
        BudgetOption.GPU_COUNT: gpus
    }
    train_job = client.create_train_job(app, task, train_dataset['id'], val_dataset['id'], 
                                        budget, models=[xgb_model['id']], train_args={ 'features': features, 'target': target })
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

    if queries is not None:
        print('Making predictions for queries:')
        print(queries)
        predictions = make_predictions(client, predictor_host, queries)
        print('Predictions are:')
        print(predictions)

    print('Stopping inference job...')
    pprint(client.stop_inference_job(app))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use for training')
    parser.add_argument('--hours', type=float, default=0.1, help='How long the train job should run for (in hours)') 
    parser.add_argument('--csv', type=str, default='https://course1.winona.edu/bdeppa/Stat%20425/Data/bodyfat.csv', help='Path to a standard CSV file to perform regression on')
    parser.add_argument('--features', type=str, default=None, help='List of feature columns\' names as comma separated values')
    parser.add_argument('--target', type=str, default=None, help='Target column\'s name')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)

    run_tabular_regression(client, args.csv, args.gpus, args.hours,
                            features=['density',
                                    'age',
                                    'weight',
                                    'height',
                                    'neck',
                                    'chest',
                                    'abdomen',
                                    'hip',
                                    'thigh',
                                    'knee',
                                    'ankle',
                                    'biceps',
                                    'forearm',
                                    'wrist'],
                            target='bodyfat',
                            queries=[
                                {'density': 1.0207,
                                'age': 65,
                                'weight': 224.5,
                                'height': 68.25,
                                'neck': 38.8,
                                'chest': 119.6,
                                'abdomen': 118.0,
                                'hip': 114.3,
                                'thigh': 61.3,
                                'knee': 42.1,
                                'ankle': 23.4,
                                'biceps': 34.9,
                                'forearm': 30.1,
                                'wrist': 19.4}
                            ])
