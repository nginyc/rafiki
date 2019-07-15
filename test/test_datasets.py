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
from test.utils import global_setup, make_app_dev, gen, DATASET_FILE_PATH

class TestDatasets():

    @pytest.fixture(scope='class')
    def app_dev_create_dataset(self):
        app_dev = make_app_dev()
        (name, task, file_path) = make_dataset_info()

        # Create dataset
        dataset = app_dev.create_dataset(name, task, file_path)
        assert 'id' in dataset
        dataset_id = dataset['id']

        return (app_dev, dataset_id, task)
    
    def test_app_dev_create_dataset(self, app_dev_create_dataset):
        (app_dev, dataset_id, task) = app_dev_create_dataset
        app_dev: Client

        # Can view created dataset
        datasets = app_dev.get_datasets()
        assert any([(x['id'] == dataset_id) for x in datasets])
        datasets = app_dev.get_datasets(task=task)
        assert any([(x['id'] == dataset_id) for x in datasets])


def make_dataset_info():
    name = gen()
    task = gen()
    file_path = DATASET_FILE_PATH
    return (name, task, file_path)

