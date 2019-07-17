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
import tempfile
import os

from rafiki.constants import ModelAccessRight
from test.utils import global_setup, make_model_dev, make_app_dev, gen, superadmin, MODEL_CLASS, MODEL_FILE_PATH


class TestModels():

    @pytest.fixture(scope='class')
    def model_dev_create_model(self):
        model_dev = make_model_dev()
        (name, task, model_file_path, model_class, dependencies) = make_model_info()
        model = model_dev.create_model(name, task, model_file_path, model_class, dependencies)
        assert 'id' in model
        model_id = model['id']

        return (model_dev, model_id, name, task, model_file_path)
    
    def test_model_dev_create_model(self, model_dev_create_model):
        (model_dev, model_id, name, task, model_file_path) = model_dev_create_model

        # Model upon getting by ID should match created model
        model = model_dev.get_model(model_id)
        assert model['id'] == model_id
        assert model['name'] == name
        assert model['task'] == task

        # Available models should contain created model
        models = model_dev.get_available_models()
        assert any([x['id'] == model_id for x in models])

    def test_model_dev_cant_view_others_model(self, model_dev_create_model):
        (model_dev, model_id, name, task, model_file_path) = model_dev_create_model
        model_dev2 = make_model_dev()

        # Model dev 2 can't view model
        with pytest.raises(Exception):
            model_dev2.get_model(model_id)
    
        # Model dev 2 doesn't see it in available models
        models = model_dev2.get_available_models()
        assert not any([x['id'] == model_id for x in models])


    def test_model_dev_download_model(self, model_dev_create_model):
        (model_dev, model_id, name, task, model_file_path) = model_dev_create_model

        # Download model file, should be the same size
        with tempfile.NamedTemporaryFile() as f:
            model_dev.download_model_file(model_id, f.name)
            assert os.stat(f.name).st_size == os.stat(model_file_path).st_size


    def test_model_dev_cant_download_others_model(self, model_dev_create_model):
        (model_dev, model_id, name, task, model_file_path) = model_dev_create_model
        model_dev2 = make_model_dev()

        # Model dev 2 can't delete model file
        with pytest.raises(Exception):
            with tempfile.NamedTemporaryFile() as f:
                model_dev2.download_model_file(model_id, f.name)


    def test_model_dev_delete_model(self, model_dev_create_model):
        (model_dev, model_id, name, task, model_file_path) = model_dev_create_model

        # Delete model
        model_dev.delete_model(model_id)

        # Can't view model
        with pytest.raises(Exception):
            model_dev.get_model(model_id)
    
        # Model not in available models
        models = model_dev.get_available_models()
        assert not any([x['id'] == model_id for x in models])


    def test_model_dev_cant_delete_others_model(self, model_dev_create_model):
        (model_dev, model_id, name, task, model_file_path) = model_dev_create_model
        model_dev2 = make_model_dev()

        # Model dev 2 can't download model file
        with pytest.raises(Exception):
            model_dev2.delete_model(model_id)


    def test_app_dev_view_public_model(self):
        app_dev = make_app_dev()
        model_dev = make_model_dev()
        (name, task, model_file_path, model_class, dependencies) = make_model_info()

        # Model dev creates public model
        model = model_dev.create_model(name, task, model_file_path, model_class, dependencies, access_right=ModelAccessRight.PUBLIC)
        assert 'id' in model
        model_id = model['id']

        # App dev sees it in available models
        models = app_dev.get_available_models()
        assert any([x['id'] == model_id for x in models])
       
        # App dev sees it in available models after filtering by task
        models = app_dev.get_available_models(task=task)
        assert any([x['id'] == model_id for x in models])


def make_model_info():
    task = gen()
    name = gen()
    model_file_path = MODEL_FILE_PATH
    model_class = MODEL_CLASS
    dependencies = {}
    return (task, name, model_file_path, model_class, dependencies)

