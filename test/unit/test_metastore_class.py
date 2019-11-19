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
import uuid
import bcrypt
from rafiki.meta_store import MetaStore
from rafiki.constants import UserType
from rafiki.data_store import DataStore
from rafiki.constants import TrainJobStatus, ModelAccessRight, ServiceType, \
    ServiceStatus, InferenceJobStatus, TrialStatus
import mock
from sqlalchemy.orm import Query


METASTORE_SETTING={
    'postgres_host': 'rafiki-db-unittest',
    'postgres_port': 30001,
    'postgres_user': 'rafiki',
    'postgres_db': 'rafiki',
    'postgres_password': 'rafiki'
}
# METASTORE_SETTING={
#     'postgres_host': 'localhost',
#     'postgres_port': 5432,
#     'postgres_user': 'rafiki',
#     'postgres_db': 'rafiki',
#     'postgres_password': 'rafiki'
# }
# rafiki/meta_store/meta_store.py
class TestMetaStore(object):
    @pytest.fixture(scope='class')
    def metastore(self):
        return MetaStore(**METASTORE_SETTING)

    def test_init(self, metastore):
        metastore_obj = metastore
        assert metastore_obj._engine != None
        assert metastore_obj._Session != None

    def test_connect(self, metastore):
        metastore_obj = metastore
        metastore_obj.connect()
        assert metastore_obj._session != None

    @pytest.fixture(scope='class', params=[UserType.ADMIN, \
        UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER,])
    def usertype(self, request):
        return request.param

    @pytest.fixture(scope='class', params=['InvalidUser'])
    def usertype_invalid(self, request):
        return request.param

    def test_create_user(self, metastore, usertype):
        metastore_obj = metastore
        email = f'{usertype}@rafiki'
        password = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = metastore_obj.create_user(email, password_hash, usertype)
        assert user.email == email
        assert user.password_hash == password_hash
        assert user.user_type == usertype

    def test_create_user_invalid(self, metastore, usertype_invalid):
        metastore_obj = metastore
        email = f'{usertype_invalid}@rafiki'
        password = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        with pytest.raises(Exception):
            user = metastore_obj.create_user(email, password_hash, usertype_invalid)
    
    def test_get_user_by_email(self, metastore, usertype):
        metastore_obj = metastore
        email = f'{usertype}@rafiki'
        user = metastore_obj.get_user_by_email(email)
        assert user.email == email
        assert user.user_type == usertype

    def test_get_user_by_email_invalid(self, metastore, usertype_invalid):
        metastore_obj = metastore
        email = f'{usertype_invalid}@rafiki'
        user = metastore_obj.get_user_by_email(email)
        assert user == None

    def test_get_users(self, metastore, usertype):
        metastore_obj = metastore
        email = f'{usertype}@rafiki'
        users = metastore_obj.get_users()
        assert any([user.email == email and user.user_type == usertype for user in users])
    
    def test_ban_user(self, metastore, usertype):
        metastore_obj = metastore
        email = f'{usertype}@rafiki'
        user = metastore_obj.get_user_by_email(email)
        metastore_obj.ban_user(user)
        user = metastore_obj.get_user_by_email(email)
        assert user.banned_date != None

    def test__validate_user_type(self, metastore, usertype_invalid):
        metastore_obj = metastore
        with pytest.raises(Exception):
            metastore_obj._validate_user_type(usertype_invalid)


    @pytest.fixture(scope='class')
    def dataset_info(self, metastore):
        metastore_obj = metastore
        email = 'dataset@rafiki'
        password = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = metastore_obj.create_user(email, password_hash, UserType.MODEL_DEVELOPER)
        metastore_obj.commit()
        return {
            'store_dataset_id': str(uuid.uuid4()),
            'name': 'test_dataset',
            'task': 'test_task',
            'size_bytes': 100,
            'owner_id': user.id
            }

    def test_create_dataset(self, metastore, dataset_info):
        metastore_obj = metastore
        dataset = metastore_obj.create_dataset(dataset_info['name'], dataset_info['task'], \
            dataset_info['size_bytes'], dataset_info['store_dataset_id'], dataset_info['owner_id'])
        assert dataset.name == dataset_info['name']
        assert dataset.task == dataset_info['task']
        assert dataset.size_bytes == dataset_info['size_bytes']
        assert dataset.owner_id == dataset_info['owner_id']
        assert dataset.store_dataset_id == dataset_info['store_dataset_id']

    def test_get_dataset(self, metastore, dataset_info):
        metastore_obj = metastore
        datasets = metastore_obj.get_datasets(dataset_info['owner_id'])
        assert len(datasets) == 1

        dataset = metastore_obj.get_dataset(datasets[0].id)
        assert dataset.name == dataset_info['name']
        assert dataset.task == dataset_info['task']
        assert dataset.size_bytes == dataset_info['size_bytes']
        assert dataset.owner_id == dataset_info['owner_id']
        assert dataset.store_dataset_id == dataset_info['store_dataset_id']

    def test_get_datasets(self, metastore, dataset_info):
        metastore_obj = metastore
        datasets = metastore_obj.get_datasets(dataset_info['owner_id'])
        assert len(datasets) == 1
        assert datasets[0].name == dataset_info['name']
        assert datasets[0].task == dataset_info['task']
        assert datasets[0].size_bytes == dataset_info['size_bytes']
        assert datasets[0].owner_id == dataset_info['owner_id']
        assert datasets[0].store_dataset_id == dataset_info['store_dataset_id']

        datasets = metastore_obj.get_datasets(dataset_info['owner_id'], dataset_info['task'])
        assert len(datasets) == 1
        assert datasets[0].name == dataset_info['name']
        assert datasets[0].task == dataset_info['task']
        assert datasets[0].size_bytes == dataset_info['size_bytes']
        assert datasets[0].owner_id == dataset_info['owner_id']
        assert datasets[0].store_dataset_id == dataset_info['store_dataset_id']

        datasets = metastore_obj.get_datasets(dataset_info['owner_id'], 'else_task')
        assert len(datasets) == 0
    
    @pytest.fixture(scope='class')
    def train_job_info(self, metastore, dataset_info):
        metastore_obj = metastore
        email = 'train@rafiki'
        password = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = metastore_obj.create_user(email, password_hash, UserType.MODEL_DEVELOPER)
        metastore_obj.commit()
        dataset = metastore_obj.create_dataset(dataset_info['name'], dataset_info['task'], \
            dataset_info['size_bytes'], dataset_info['store_dataset_id'], user.id)
        metastore_obj.commit()
        return {
            'user_id': user.id,
            'app': 'test_train',
            'app_version': 1,
            'task': 'test',
            'budget': {'GPU_COUNT': 0},
            'train_dataset_id': dataset.id,
            'val_dataset_id': dataset.id,
            'train_args': {}
        }

    def test_create_train_job(self, metastore, train_job_info):
        metastore_obj = metastore
        train_job = metastore.create_train_job(train_job_info['user_id'], train_job_info['app'], \
            train_job_info['app_version'], train_job_info['task'], train_job_info['budget'], \
                train_job_info['train_dataset_id'], train_job_info['val_dataset_id'], train_job_info['train_args'])
        metastore_obj.commit()
        assert train_job.user_id == train_job_info['user_id']
        assert train_job.app == train_job_info['app']
        assert train_job.app_version == train_job_info['app_version']
        assert train_job.task == train_job_info['task']
        assert train_job.budget == train_job_info['budget']
        assert train_job.train_dataset_id == train_job_info['train_dataset_id']
        assert train_job.val_dataset_id == train_job_info['val_dataset_id']
        assert train_job.train_args == train_job_info['train_args']
    
    def test_get_train_jobs_by_app(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_app(train_job_info['user_id'], train_job_info['app'])
        assert len(train_jobs) == 1
        assert train_jobs[0].user_id == train_job_info['user_id']
        assert train_jobs[0].app == train_job_info['app']
        assert train_jobs[0].app_version == train_job_info['app_version']
        assert train_jobs[0].task == train_job_info['task']
        assert train_jobs[0].budget == train_job_info['budget']
        assert train_jobs[0].train_dataset_id == train_job_info['train_dataset_id']
        assert train_jobs[0].val_dataset_id == train_job_info['val_dataset_id']
        assert train_jobs[0].train_args == train_job_info['train_args']

    def test_get_train_jobs_by_user(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert len(train_jobs) == 1
        assert train_jobs[0].user_id == train_job_info['user_id']
        assert train_jobs[0].app == train_job_info['app']
        assert train_jobs[0].app_version == train_job_info['app_version']
        assert train_jobs[0].task == train_job_info['task']
        assert train_jobs[0].budget == train_job_info['budget']
        assert train_jobs[0].train_dataset_id == train_job_info['train_dataset_id']
        assert train_jobs[0].val_dataset_id == train_job_info['val_dataset_id']
        assert train_jobs[0].train_args == train_job_info['train_args']

    def test_get_train_job(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert len(train_jobs) == 1

        train_job = metastore_obj.get_train_job(train_jobs[0].id)
        assert train_job.user_id == train_job_info['user_id']
        assert train_job.app == train_job_info['app']
        assert train_job.app_version == train_job_info['app_version']
        assert train_job.task == train_job_info['task']
        assert train_job.budget == train_job_info['budget']
        assert train_job.train_dataset_id == train_job_info['train_dataset_id']
        assert train_job.val_dataset_id == train_job_info['val_dataset_id']
        assert train_job.train_args == train_job_info['train_args']

    def test_get_train_jobs_by_statuses(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_statuses([TrainJobStatus.STARTED])
        assert len(train_jobs) == 1
        assert train_jobs[0].user_id == train_job_info['user_id']
        assert train_jobs[0].app == train_job_info['app']
        assert train_jobs[0].app_version == train_job_info['app_version']
        assert train_jobs[0].task == train_job_info['task']
        assert train_jobs[0].budget == train_job_info['budget']
        assert train_jobs[0].train_dataset_id == train_job_info['train_dataset_id']
        assert train_jobs[0].val_dataset_id == train_job_info['val_dataset_id']
        assert train_jobs[0].train_args == train_job_info['train_args']

        train_jobs = metastore_obj.get_train_jobs_by_statuses([TrainJobStatus.STOPPED])
        assert len(train_jobs) == 0

    def test_get_train_job_by_app_version(self, metastore, train_job_info):
        metastore_obj = metastore
        train_job = metastore_obj.get_train_job_by_app_version(train_job_info['user_id'], \
            train_job_info['app'], train_job_info['app_version'])
        assert train_job.user_id == train_job_info['user_id']
        assert train_job.app == train_job_info['app']
        assert train_job.app_version == train_job_info['app_version']
        assert train_job.task == train_job_info['task']
        assert train_job.budget == train_job_info['budget']
        assert train_job.train_dataset_id == train_job_info['train_dataset_id']
        assert train_job.val_dataset_id == train_job_info['val_dataset_id']
        assert train_job.train_args == train_job_info['train_args']
    
    def test_mark_train_job_as_running(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert len(train_jobs) == 1

        train_job = train_jobs[0]
        metastore_obj.mark_train_job_as_running(train_job)
        assert train_job.status == TrainJobStatus.RUNNING

        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert train_jobs[0].status == TrainJobStatus.RUNNING

    def test_mark_train_job_as_errored(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert len(train_jobs) == 1

        train_job = train_jobs[0]
        metastore_obj.mark_train_job_as_errored(train_job)
        assert train_job.status == TrainJobStatus.ERRORED
        
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert train_jobs[0].status == TrainJobStatus.ERRORED

    def test_mark_train_job_as_stopped(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert len(train_jobs) == 1

        train_job = train_jobs[0]
        metastore_obj.mark_train_job_as_stopped(train_job)
        assert train_job.status == TrainJobStatus.STOPPED
        
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert train_jobs[0].status == TrainJobStatus.STOPPED

    

    @pytest.fixture(scope='class')
    def model_info(self, metastore):
        metastore_obj = metastore
        email = 'model@rafiki'
        password = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = metastore_obj.create_user(email, password_hash, UserType.MODEL_DEVELOPER)
        metastore_obj.commit()
        with open('test/data/Model.py', 'rb') as f:
            model_file_bytes = f.read()
        return {
            'user_id': user.id,
            'name': 'test_model',
            'task': 'test',
            'model_file_bytes': model_file_bytes,
            'model_class': 'Model',
            'dependencies': {} 
        }

    def test_create_model(self, metastore, model_info):
        metastore_obj = metastore
        model = metastore_obj.create_model(model_info['user_id'], model_info['name'], \
            model_info['task'], model_info['model_file_bytes'], model_info['model_class'], \
                'image', model_info['dependencies'], ModelAccessRight.PUBLIC)
        metastore_obj.commit()
        assert model.user_id == model_info['user_id']
        assert model.name == model_info['name']
        assert model.task == model_info['task']
        assert model.model_file_bytes == model_info['model_file_bytes']
        assert model.model_class == model_info['model_class']
        assert model.dependencies == model_info['dependencies']

    def test_get_available_models(self, metastore, model_info):
        metastore_obj = metastore
        models = metastore_obj.get_available_models(model_info['user_id'])
        assert len(models) == 1
        assert models[0].user_id == model_info['user_id']
        assert models[0].name == model_info['name']
        assert models[0].task == model_info['task']
        assert models[0].model_file_bytes == model_info['model_file_bytes']
        assert models[0].model_class == model_info['model_class']
        assert models[0].dependencies == model_info['dependencies']
        
    def test_get_model_by_name(self, metastore, model_info):
        metastore_obj = metastore
        model = metastore_obj.get_model_by_name(model_info['user_id'], model_info['name'])
        assert model.user_id == model_info['user_id']
        assert model.name == model_info['name']
        assert model.task == model_info['task']
        assert model.model_file_bytes == model_info['model_file_bytes']
        assert model.model_class == model_info['model_class']
        assert model.dependencies == model_info['dependencies']
        model = metastore_obj.get_model_by_name(model_info['user_id'], 'InvalidModel')
        assert model is None

    def test_get_model(self, metastore, model_info):
        metastore_obj = metastore
        models = metastore_obj.get_available_models(model_info['user_id'])
        assert len(models) == 1

        model = metastore_obj.get_model(models[0].id)
        assert model.user_id == model_info['user_id']
        assert model.name == model_info['name']
        assert model.task == model_info['task']
        assert model.model_file_bytes == model_info['model_file_bytes']
        assert model.model_class == model_info['model_class']
        assert model.dependencies == model_info['dependencies']
    
    @mock.patch('sqlalchemy.orm.Query.first')
    def test_delete_model(self, mock_first, metastore, model_info):
        metastore_obj = metastore
        models = metastore_obj.get_available_models(model_info['user_id'])
        assert len(models) == 1

        mock_first.return_value = 1
        # Query.first = mock.Mock(return_value=1)
        with pytest.raises(Exception):
            metastore_obj.delete_model(models[0])

        mock_first.return_value = None
        # Query.first = mock.Mock(return_value=None)
        metastore_obj.delete_model(models[0])
        metastore_obj.commit()
        
        models = metastore_obj.get_available_models(model_info['user_id'])
        assert len(models) == 0
    
    # @pytest.fixture(scope='class', params=[ServiceType.TRAIN, ServiceType.ADVISOR, \
    #     ServiceType.INFERENCE, ServiceType.PREDICT])
    # def servicetype(self, request):
    #     return request.param

    @pytest.fixture(scope='class')
    def service_info(self):
        return {
            'service_type': ServiceType.TRAIN, 
            'container_manager_type': 'kubernetes',
            'docker_image': ServiceType.TRAIN+'_image', 
            'replicas': 1,
            'gpus': 0
        }

    def test_create_service(self, metastore, service_info):
        metastore_obj = metastore
        service = metastore_obj.create_service(service_info['service_type'], service_info['container_manager_type'], \
            service_info['docker_image'], service_info['replicas'], service_info['gpus'])
        assert service.service_type == service_info['service_type']
        assert service.container_manager_type == service_info['container_manager_type']
        assert service.docker_image == service_info['docker_image']
        assert service.replicas == service_info['replicas']
        assert service.gpus == service_info['gpus']
    
    def test_get_services(self, metastore, service_info):
        metastore_obj = metastore
        services = metastore_obj.get_services()
        assert len(services) == 1
        assert services[0].service_type == service_info['service_type']
        assert services[0].container_manager_type == service_info['container_manager_type']
        assert services[0].docker_image == service_info['docker_image']
        assert services[0].replicas == service_info['replicas']
        assert services[0].gpus == service_info['gpus']

    def test_get_service(self, metastore, service_info):
        metastore_obj = metastore
        services = metastore_obj.get_services()
        assert len(services) == 1
        service = metastore_obj.get_service(services[0].id)
        assert service.service_type == service_info['service_type']
        assert service.container_manager_type == service_info['container_manager_type']
        assert service.docker_image == service_info['docker_image']
        assert service.replicas == service_info['replicas']
        assert service.gpus == service_info['gpus']

    def test_mark_service_as_deploying(self, metastore):
        metastore_obj = metastore
        services = metastore_obj.get_services()
        assert len(services) == 1
        metastore_obj.mark_service_as_deploying(services[0], 'train_service_name', 'train_service_id', \
            'test_host', 1001, '255.255.255.255', 1001, {})
        metastore_obj.commit()
        services = metastore_obj.get_services()
        assert len(services) == 1
        assert services[0].status == ServiceStatus.DEPLOYING

    def test_mark_service_as_running(self, metastore):
        metastore_obj = metastore
        services = metastore_obj.get_services()
        assert len(services) == 1
        metastore_obj.mark_service_as_running(services[0])
        metastore_obj.commit()
        services = metastore_obj.get_services()
        assert len(services) == 1
        assert services[0].status == ServiceStatus.RUNNING

    def test_mark_service_as_errored(self, metastore):
        metastore_obj = metastore
        services = metastore_obj.get_services()
        assert len(services) == 1
        metastore_obj.mark_service_as_errored(services[0])
        metastore_obj.commit()
        services = metastore_obj.get_services()
        assert len(services) == 1
        assert services[0].status == ServiceStatus.ERRORED
    
    def test_mark_service_as_stopped(self, metastore):
        metastore_obj = metastore
        services = metastore_obj.get_services()
        assert len(services) == 1
        metastore_obj.mark_service_as_stopped(services[0])
        metastore_obj.commit()
        services = metastore_obj.get_services()
        assert len(services) == 1
        assert services[0].status == ServiceStatus.STOPPED

    @pytest.fixture(scope='class')
    def sub_train_job_info(self, metastore, model_info, train_job_info):
        model = metastore.create_model(model_info['user_id'], 'model_for_sub_train_job', \
            model_info['task'], model_info['model_file_bytes'], model_info['model_class'], \
                'image', model_info['dependencies'], ModelAccessRight.PUBLIC)
        metastore.commit()
        assert model.user_id == model_info['user_id']
        assert model.name == 'model_for_sub_train_job'
        assert model.task == model_info['task']
        assert model.model_file_bytes == model_info['model_file_bytes']
        assert model.model_class == model_info['model_class']
        assert model.dependencies == model_info['dependencies']

        train_jobs = metastore.get_train_jobs_by_user(train_job_info['user_id'])
        assert len(train_jobs) == 1

        service = metastore.create_service(
            service_type='ADVISOR',
            container_manager_type='Kubernetes',
            docker_image='rafiki-worker',
            replicas=1,
            gpus=0
        )
        metastore.commit()
        assert service.service_type == 'ADVISOR'
        assert service.container_manager_type == 'Kubernetes'
        assert service.docker_image == 'rafiki-worker'
        assert service.replicas == 1
        assert service.gpus == 0

        return {
            'model_id': model.id,
            'train_job_id': train_jobs[0].id, 
            'advisor_service_id': service.id
        }

    def test_create_sub_train_job(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_job = metastore_obj.create_sub_train_job(sub_train_job_info['train_job_id'], \
            sub_train_job_info['model_id'])
        metastore_obj.commit()
        assert sub_train_job.train_job_id == sub_train_job_info['train_job_id']
        assert sub_train_job.model_id == sub_train_job_info['model_id']
       
    def test_get_sub_train_jobs_of_train_job(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        assert sub_train_jobs[0].train_job_id == sub_train_job_info['train_job_id']
        assert sub_train_jobs[0].model_id == sub_train_job_info['model_id']

        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job('InvalidTrainJobId')
        assert len(sub_train_jobs) == 0

    def test_update_sub_train_job(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        sub_train_job = metastore_obj.update_sub_train_job(sub_train_jobs[0], sub_train_job_info['advisor_service_id'])
        metastore_obj.commit()
        assert sub_train_job.advisor_service_id == sub_train_job_info['advisor_service_id']

    def test_get_sub_train_job_by_advisor(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_job = metastore_obj.get_sub_train_job_by_advisor(sub_train_job_info['advisor_service_id'])
        assert sub_train_job.train_job_id == sub_train_job_info['train_job_id']
        assert sub_train_job.model_id == sub_train_job_info['model_id']
        assert sub_train_job.advisor_service_id == sub_train_job_info['advisor_service_id']

    def test_get_sub_train_job(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        sub_train_job = metastore_obj.get_sub_train_job(sub_train_jobs[0].id)
        assert sub_train_job.id == sub_train_jobs[0].id
        assert sub_train_job.train_job_id == sub_train_jobs[0].train_job_id
        assert sub_train_job.model_id == sub_train_jobs[0].model_id

    def test_mark_sub_train_job_as_running(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        metastore_obj.mark_sub_train_job_as_running(sub_train_jobs[0])
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert sub_train_jobs[0].status == TrainJobStatus.RUNNING

    def test_mark_sub_train_job_as_errored(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        metastore_obj.mark_sub_train_job_as_errored(sub_train_jobs[0])
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert sub_train_jobs[0].status == TrainJobStatus.ERRORED

    def test_mark_sub_train_job_as_stopped(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        metastore_obj.mark_sub_train_job_as_stopped(sub_train_jobs[0])
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert sub_train_jobs[0].status == TrainJobStatus.STOPPED

    @pytest.fixture(scope='class')
    def train_service_info(self, metastore):
        service = metastore.create_service(ServiceType.TRAIN, 'test_type', 'test_train_image', 1, 0)
        metastore.commit()
        return {
            'service_id': service.id
        }

    def test_create_train_job_worker(self, metastore, sub_train_job_info, train_service_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        train_job_worker = metastore_obj.create_train_job_worker(train_service_info['service_id'], sub_train_jobs[0].id)
        assert train_job_worker.service_id == train_service_info['service_id']
        assert train_job_worker.sub_train_job_id == sub_train_jobs[0].id

    def test_get_train_job_worker(self, metastore, train_service_info):
        metastore_obj = metastore
        train_job_worker = metastore_obj.get_train_job_worker(train_service_info['service_id'])
        assert train_job_worker.service_id == train_service_info['service_id']

    def test_get_workers_of_sub_train_job(self, metastore, sub_train_job_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        train_job_workers = metastore_obj.get_workers_of_sub_train_job(sub_train_jobs[0].id)
        assert len(train_job_workers) == 1
        assert train_job_workers[0].sub_train_job_id == sub_train_jobs[0].id
    
    @pytest.fixture(scope='class')
    def inference_job_info(self, metastore, train_job_info):
        metastore_obj = metastore
        train_jobs = metastore_obj.get_train_jobs_by_user(train_job_info['user_id'])
        assert len(train_jobs) == 1
        service = metastore.create_service(ServiceType.PREDICT, 'test_type', 'test_predictor_image', 1, 0)
        metastore.commit()
        return {
            'user_id': train_job_info['user_id'],
            'train_job_id': train_jobs[0].id,
            'budget': {'GPU_COUNT': 1},
            'predictor_service_id': service.id
            }
        
    def test_create_inference_job(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_job = metastore_obj.create_inference_job(inference_job_info['user_id'], \
            inference_job_info['train_job_id'], inference_job_info['budget'])
        metastore_obj.commit()
        assert inference_job.user_id == inference_job_info['user_id']
        assert inference_job.train_job_id == inference_job_info['train_job_id']
        assert inference_job.budget == inference_job_info['budget']

    def test_get_inference_jobs_by_user(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        assert inference_jobs[0].user_id == inference_job_info['user_id']

    def test_get_inference_job(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        assert inference_jobs[0].user_id == inference_job_info['user_id']
        inference_job = metastore_obj.get_inference_job(inference_jobs[0].id)
        assert inference_job.user_id == inference_job_info['user_id']
        assert inference_job.train_job_id == inference_job_info['train_job_id']
        assert inference_job.budget == inference_job_info['budget']

    def test_update_inference_job(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        inference_job = metastore_obj.update_inference_job(inference_jobs[0], inference_job_info['predictor_service_id'])
        assert inference_job.predictor_service_id == inference_job_info['predictor_service_id']

    def test_mark_inference_job_as_errored(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        metastore_obj.mark_inference_job_as_errored(inference_jobs[0])
        metastore_obj.commit()
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        assert inference_jobs[0].status == InferenceJobStatus.ERRORED

    def test_mark_inference_job_as_stopped(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        metastore_obj.mark_inference_job_as_stopped(inference_jobs[0])
        metastore_obj.commit()
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        assert inference_jobs[0].status == InferenceJobStatus.STOPPED

    def test_mark_inference_job_as_running(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        metastore_obj.mark_inference_job_as_running(inference_jobs[0])
        metastore_obj.commit()
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        assert inference_jobs[0].status == InferenceJobStatus.RUNNING

    def test_get_deployed_inference_job_by_train_job(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_job = metastore_obj.get_deployed_inference_job_by_train_job(inference_job_info['train_job_id'])
        assert inference_job.status == InferenceJobStatus.STARTED or inference_job.status == InferenceJobStatus.RUNNING
        metastore_obj.mark_inference_job_as_stopped(inference_job)
        metastore_obj.commit()
        inference_job = metastore_obj.get_deployed_inference_job_by_train_job(inference_job_info['train_job_id'])
        assert inference_job is None
    
    def test_get_inference_jobs_by_statuses(self, metastore, inference_job_info):
        metastore_obj = metastore
        inference_jobs = metastore_obj.get_inference_jobs_by_statuses([InferenceJobStatus.STOPPED])
        assert len(inference_jobs) == 1
        assert inference_jobs[0].status == InferenceJobStatus.STOPPED
    
    @pytest.fixture(scope='class')
    def trial_info(self, metastore, sub_train_job_info, train_service_info):
        metastore_obj = metastore
        sub_train_jobs = metastore_obj.get_sub_train_jobs_of_train_job(sub_train_job_info['train_job_id'])
        assert len(sub_train_jobs) == 1
        return {
            'sub_train_job_id': sub_train_jobs[0].id,
            'model_id': sub_train_job_info['model_id'],
            'worker_id': train_service_info['service_id'],
            'no': 1,
            'train_job_id': sub_train_job_info['train_job_id']
        }
        
    def test_create_trial(self, metastore, trial_info):
        metastore_obj = metastore
        trial_job = metastore_obj.create_trial(trial_info['sub_train_job_id'], trial_info['no'], \
            trial_info['model_id'], trial_info['worker_id'])
        assert trial_job.worker_id == trial_info['worker_id']
        assert trial_job.sub_train_job_id == trial_info['sub_train_job_id']
        assert trial_job.no == trial_info['no']
        assert trial_job.model_id == trial_info['model_id']

    def test_get_trials_of_train_job(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        assert trial_infos[0].worker_id == trial_info['worker_id']
        assert trial_infos[0].sub_train_job_id == trial_info['sub_train_job_id']
        assert trial_infos[0].no == trial_info['no']
        assert trial_infos[0].model_id == trial_info['model_id']

    def test_get_trials_of_sub_train_job(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_sub_train_job(trial_info['sub_train_job_id'])
        assert len(trial_infos) == 1
        assert trial_infos[0].worker_id == trial_info['worker_id']
        assert trial_infos[0].sub_train_job_id == trial_info['sub_train_job_id']
        assert trial_infos[0].no == trial_info['no']
        assert trial_infos[0].model_id == trial_info['model_id']

    def test_get_trial(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        trial_job = metastore_obj.get_trial(trial_infos[0].id)
        assert trial_job.worker_id == trial_info['worker_id']
        assert trial_job.sub_train_job_id == trial_info['sub_train_job_id']
        assert trial_job.no == trial_info['no']
        assert trial_job.model_id == trial_info['model_id']

    def test_mark_trial_as_running(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        metastore_obj.mark_trial_as_running(trial_infos[0], {})
        metastore_obj.commit()
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        assert trial_infos[0].status == TrialStatus.RUNNING

    def test_mark_trial_as_errored(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        metastore_obj.mark_trial_as_errored(trial_infos[0])
        metastore_obj.commit()
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        assert trial_infos[0].status == TrialStatus.ERRORED
    
    def test_mark_trial_as_completed(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        metastore_obj.mark_trial_as_completed(trial_infos[0], 90.1, 'test_store_id')
        metastore_obj.commit()
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        assert trial_infos[0].status == TrialStatus.COMPLETED

    def test_get_best_trials_of_train_job(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_best_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        assert trial_infos[0].worker_id == trial_info['worker_id']
        assert trial_infos[0].sub_train_job_id == trial_info['sub_train_job_id']
        assert trial_infos[0].no == trial_info['no']
        assert trial_infos[0].model_id == trial_info['model_id']

    def test_get_best_trials_of_sub_train_job(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_best_trials_of_sub_train_job(trial_info['sub_train_job_id'])
        assert len(trial_infos) == 1
        assert trial_infos[0].worker_id == trial_info['worker_id']
        assert trial_infos[0].sub_train_job_id == trial_info['sub_train_job_id']
        assert trial_infos[0].no == trial_info['no']
        assert trial_infos[0].model_id == trial_info['model_id']
    
    def test_add_trial_log(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        trial_log = metastore_obj.add_trial_log(trial_infos[0], 'test_log', 'info')
        metastore_obj.commit()
        assert trial_log.trial_id == trial_infos[0].id
        assert trial_log.line == 'test_log'
        assert trial_log.level == 'info'
    
    def test_get_trial_logs(self, metastore, trial_info):
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        trial_logs = metastore_obj.get_trial_logs(trial_infos[0].id)
        assert len(trial_logs) == 1
        assert trial_logs[0].line == 'test_log'
        assert trial_logs[0].level == 'info'
    
    @pytest.fixture(scope='class')
    def inference_job_worker_info(self, metastore, inference_job_info, trial_info):
        service = metastore.create_service(ServiceType.INFERENCE, 'test_type', 'test_inference_image', 1, 0)
        metastore.commit()
        metastore_obj = metastore
        trial_infos = metastore_obj.get_trials_of_train_job(trial_info['train_job_id'])
        assert len(trial_infos) == 1
        inference_jobs = metastore_obj.get_inference_jobs_by_user(inference_job_info['user_id'])
        assert len(inference_jobs) == 1
        return {
            'service_id': service.id,
            'inference_job_id': inference_jobs[0].id,
            'trial_id': trial_infos[0].id 
        }

    def test_create_inference_job_worker(self, metastore, inference_job_worker_info):
        metastore_obj = metastore
        worker = metastore_obj.create_inference_job_worker(inference_job_worker_info['service_id'], \
            inference_job_worker_info['inference_job_id'], inference_job_worker_info['trial_id'])
        metastore_obj.commit()
        assert worker.service_id == inference_job_worker_info['service_id']
        assert worker.inference_job_id == inference_job_worker_info['inference_job_id']
        assert worker.trial_id == inference_job_worker_info['trial_id']

    def test_get_inference_job_worker(self, metastore, inference_job_worker_info):
        metastore_obj = metastore
        worker = metastore_obj.get_inference_job_worker(inference_job_worker_info['service_id'])
        assert worker.service_id == inference_job_worker_info['service_id']

    def test_get_workers_of_inference_job(self, metastore, inference_job_worker_info):
        metastore_obj = metastore
        workers = metastore_obj.get_workers_of_inference_job(inference_job_worker_info['inference_job_id'])
        assert len(workers) == 1
        assert workers[0].inference_job_id == inference_job_worker_info['inference_job_id']

    def test_disconnect(self, metastore):
        metastore_obj = metastore
        metastore_obj.disconnect()
        assert metastore_obj._session == None
    