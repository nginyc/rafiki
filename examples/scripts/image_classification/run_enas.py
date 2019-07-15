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

import os
import argparse
from pprint import pprint

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import BudgetOption, ModelDependency

from examples.scripts.utils import gen_id
from examples.datasets.image_files.load_cifar10 import load_cifar10

def run_enas(client, train_dataset_path, val_dataset_path, gpus, hours):    
    '''
        Conducts training of model `TfEnas` on the CIFAR-10 dataset for IMAGE_CLASSIFICATION.
        Demonstrates architecture tuning with ENAS on Rafiki. 
    '''
    task = 'IMAGE_CLASSIFICATION'

    app_id = gen_id()
    app = 'cifar10_enas_{}'.format(app_id)
    model_name = 'TfEnas_{}'.format(app_id)

    print('Preprocessing datasets...')
    load_cifar10(train_dataset_path, val_dataset_path)

    print('Creating & uploading datasets onto Rafiki...')
    train_dataset = client.create_dataset('{}_train'.format(app), task, train_dataset_path)
    pprint(train_dataset)
    val_dataset = client.create_dataset('{}_val'.format(app), task, val_dataset_path)
    pprint(val_dataset)

    print('Creating model...')
    model = client.create_model(
        name=model_name,
        task='IMAGE_CLASSIFICATION',
        model_file_path='examples/models/image_classification/TfEnas.py',
        model_class='TfEnas',
        dependencies={ModelDependency.TENSORFLOW: '1.12.0'}
    )
    pprint(model)

    print('Creating train job...')
    budget = { 
        BudgetOption.TIME_HOURS: hours,
        BudgetOption.GPU_COUNT: gpus
    }
    train_job = client.create_train_job(app, task, train_dataset['id'], val_dataset['id'], budget, models=[model['id']])
    pprint(train_job)

    print('Monitor the train job on Rafiki Web Admin')

    # TODO: Evaluate on test dataset?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use')
    parser.add_argument('--hours', type=float, default=24, help='How long the train job should run for (in hours)') 
    out_train_dataset_path = 'data/cifar10_train.zip'
    out_val_dataset_path = 'data/cifar10_val.zip'
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)

    run_enas(client, out_train_dataset_path, out_val_dataset_path, args.gpus, args.hours)