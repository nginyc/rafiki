import pytest
import tempfile
import os

from rafiki.client import Client
from test.utils import make_model_dev, make_app_dev, gen, superadmin, DATASET_FILE_PATH


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

