from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, ForeignKey, Integer, Binary, DateTime
from sqlalchemy.dialects.postgresql import JSON, ARRAY
import uuid
from datetime import datetime

from rafiki.constants import InferenceJobStatus, ServiceStatus, TrainJobStatus, \
    TrialStatus, ModelAccessRight

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
    status = Column(String, nullable=False, default=InferenceJobStatus.STARTED)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    predictor_service_id = Column(String, ForeignKey('service.id'))
    datetime_stopped = Column(DateTime, default=None)

class InferenceJobWorker(Base):
    __tablename__ = 'inference_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    inference_job_id = Column(String, ForeignKey('inference_job.id'))
    trial_id = Column(String, ForeignKey('trial.id'), nullable=False)

class Model(Base):
    __tablename__ = 'model'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)
    name = Column(String, unique=True, nullable=False)
    task = Column(String, nullable=False)
    model_file_bytes = Column(Binary, nullable=False)
    model_class = Column(String, nullable=False)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    docker_image = Column(String, nullable=False)
    dependencies = Column(JSON, nullable=False)
    access_right = Column(String, nullable=False, default=ModelAccessRight.PRIVATE)

class Service(Base):
    __tablename__ = 'service'

    id = Column(String, primary_key=True, default=generate_uuid)
    service_type = Column(String, nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_stopped = Column(DateTime, default=None)
    status = Column(String, nullable=False, default=ServiceStatus.STARTED)
    docker_image = Column(String, nullable=False)
    container_manager_type = Column(String, nullable=False)
    replicas = Column(Integer, default=0)
    ext_hostname = Column(String)
    ext_port = Column(Integer)
    hostname = Column(String)
    port = Column(Integer)
    container_service_name = Column(String)
    container_service_id = Column(String)
    requirements = Column(ARRAY(String))

class TrainJob(Base):
    __tablename__ = 'train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    app = Column(String, nullable=False)
    app_version = Column(Integer, nullable=False)
    task = Column(String, nullable=False)
    budget = Column(JSON, nullable=False)
    train_dataset_uri = Column(String, nullable=False)
    test_dataset_uri = Column(String, nullable=False)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)

class SubTrainJob(Base):
    __tablename__ = 'sub_train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    model_id = Column(String, ForeignKey('model.id'))
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    status = Column(String, nullable=False, default=TrainJobStatus.STARTED)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_stopped = Column(DateTime, default=None)

class TrainJobWorker(Base):
    __tablename__ = 'train_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    sub_train_job_id = Column(String, ForeignKey('sub_train_job.id'), nullable=False)

class Trial(Base):
    __tablename__ = 'trial'

    id = Column(String, primary_key=True, default=generate_uuid)
    sub_train_job_id = Column(String, ForeignKey('sub_train_job.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    status = Column(String, nullable=False, default=TrialStatus.STARTED)
    knobs = Column(JSON, default=None)
    score = Column(Float, default=0)
    params_file_path = Column(String, default=None)
    datetime_stopped = Column(DateTime, default=None)

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
    password_hash = Column(Binary, nullable=False)
    user_type = Column(String, nullable=False)
    banned_date = Column(DateTime, default=None)
    