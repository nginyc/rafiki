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

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, String, Float, ForeignKey, Integer, LargeBinary, DateTime, UniqueConstraint, BigInteger
from sqlalchemy.dialects.postgresql import JSON, ARRAY
import uuid
from datetime import datetime

from rafiki.constants import InferenceJobStatus, ServiceStatus, TrainJobStatus, TrialStatus, ModelAccessRight

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

def generate_datetime():
    return datetime.utcnow()

class InferenceJob(Base):
    __tablename__ = 'inference_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    budget = Column(JSON, default={})
    status = Column(String, nullable=False, default=InferenceJobStatus.STARTED)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    predictor_service_id = Column(String, ForeignKey('service.id'))
    datetime_stopped = Column(DateTime, default=None)

class InferenceJobWorker(Base):
    __tablename__ = 'inference_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    inference_job_id = Column(String, ForeignKey('inference_job.id'))
    trial_id = Column(String, ForeignKey('trial.id'), nullable=False)

class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    task = Column(String, nullable=False)
    store_dataset_id = Column(String, nullable=False)
    size_bytes = Column(BigInteger, default=0)
    owner_id = Column(String, ForeignKey('user.id'), nullable=False)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)

class Model(Base):
    __tablename__ = 'model'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    name = Column(String, nullable=False)
    task = Column(String, nullable=False)
    model_file_bytes = Column(LargeBinary, nullable=False)
    model_class = Column(String, nullable=False)
    docker_image = Column(String, nullable=False)
    dependencies = Column(JSON, nullable=False)
    access_right = Column(String, nullable=False, default=ModelAccessRight.PRIVATE)
    __table_args__ = (UniqueConstraint('name', 'user_id'),)

class Service(Base):
    __tablename__ = 'service'

    id = Column(String, primary_key=True, default=generate_uuid)
    service_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default=ServiceStatus.STARTED)
    docker_image = Column(String, nullable=False)
    container_manager_type = Column(String, nullable=False)
    replicas = Column(Integer, nullable=False)
    gpus = Column(Integer, nullable=False)
    ext_hostname = Column(String)
    ext_port = Column(Integer)
    hostname = Column(String)
    port = Column(Integer)
    service_info = Column(JSON)
    container_service_name = Column(String)
    container_service_id = Column(String)
    container_service_info = Column(JSON)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_stopped = Column(DateTime, default=None)

    @hybrid_property
    def host(self):
        if self.ext_hostname is None or self.ext_port is None:
            return None
        return f'{self.ext_hostname}:{self.ext_port}'

class TrainJob(Base):
    __tablename__ = 'train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    app = Column(String, nullable=False)
    app_version = Column(Integer, nullable=False)
    task = Column(String, nullable=False)
    budget = Column(JSON, nullable=False)
    train_dataset_id = Column(String, ForeignKey('dataset.id'), nullable=False)
    val_dataset_id = Column(String, ForeignKey('dataset.id'), nullable=False)
    train_args = Column(JSON, default=None)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    status = Column(String, nullable=False, default=TrainJobStatus.STARTED)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_stopped = Column(DateTime, default=None)
    __table_args__ = (UniqueConstraint('app', 'app_version', 'user_id'),)

class SubTrainJob(Base):
    __tablename__ = 'sub_train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    model_id = Column(String, ForeignKey('model.id'))
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    status = Column(String, nullable=False, default=TrainJobStatus.STARTED)
    datetime_stopped = Column(DateTime, default=None)
    advisor_service_id = Column(String, ForeignKey('service.id'))

class TrainJobWorker(Base):
    __tablename__ = 'train_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    sub_train_job_id = Column(String, ForeignKey('sub_train_job.id'), nullable=False)


class Trial(Base):
    __tablename__ = 'trial'

    id = Column(String, primary_key=True, default=generate_uuid)
    no = Column(Integer, nullable=False)
    sub_train_job_id = Column(String, ForeignKey('sub_train_job.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_updated = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_stopped = Column(DateTime, default=None)
    status = Column(String, nullable=False, default=TrialStatus.PENDING)
    worker_id = Column(String, nullable=False)
    knobs = Column(JSON, default=None)
    score = Column(Float, default=None)
    store_params_id = Column(String, default=None)
    proposal = Column(JSON, default=None)

    @hybrid_property
    def is_params_saved(self):
        return self.store_params_id is not None

    __table_args__ = (UniqueConstraint('sub_train_job_id', 'no', name='_sub_train_job_id_no_uc'),) # Unique by (sub train job, trial no)

class TrialLog(Base):
    __tablename__ = 'trial_log'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime = Column(DateTime, default=generate_datetime)
    trial_id = Column(String, ForeignKey('trial.id'), nullable=False, index=True)
    line = Column(String, nullable=False)
    level = Column(String)

class User(Base):
    __tablename__ = 'user'

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(LargeBinary, nullable=False)
    user_type = Column(String, nullable=False)
    banned_date = Column(DateTime, default=None)
    