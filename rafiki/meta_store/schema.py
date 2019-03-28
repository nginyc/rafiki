from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, ForeignKey, Integer, LargeBinary, DateTime, UniqueConstraint
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
    status = Column(String, nullable=False, default=InferenceJobStatus.STARTED)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    predictor_service_id = Column(String, ForeignKey('service.id'))
    datetime_stopped = Column(DateTime, default=None)

class SubInferenceJob(Base):
    __tablename__ = 'sub_inference_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    inference_job_id = Column(String, ForeignKey('inference_job.id'))
    trial_id = Column(String, ForeignKey('trial.id'), nullable=False)
    service_id = Column(String, ForeignKey('service.id'), default=None)

class Model(Base):
    __tablename__ = 'model'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)
    name = Column(String, unique=True, nullable=False)
    task = Column(String, nullable=False)
    model_file_bytes = Column(LargeBinary, nullable=False)
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
    service_info = Column(JSON)
    container_service_name = Column(String)
    container_service_id = Column(String) # Corresponding service ID for container manager
    requirements = Column(ARRAY(String))

class TrainJob(Base):
    __tablename__ = 'train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    app = Column(String, nullable=False)
    app_version = Column(Integer, nullable=False)
    task = Column(String, nullable=False)
    budget = Column(JSON, nullable=False)
    status = Column(String, nullable=False, default=TrainJobStatus.STARTED)
    train_dataset_uri = Column(String, nullable=False)
    val_dataset_uri = Column(String, nullable=False)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    datetime_stopped = Column(DateTime, default=None)

class SubTrainJob(Base):
    __tablename__ = 'sub_train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    model_id = Column(String, ForeignKey('model.id'))
    service_id = Column(String, ForeignKey('service.id'), default=None)
    config = Column(JSON, nullable=False)

class Trial(Base):
    __tablename__ = 'trial'

    id = Column(String, primary_key=True, default=generate_uuid)
    no = Column(Integer, nullable=False)
    sub_train_job_id = Column(String, ForeignKey('sub_train_job.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_updated = Column(DateTime, nullable=False, default=generate_datetime)
    status = Column(String, nullable=False, default=TrialStatus.STARTED)
    worker_id = Column(String, nullable=False)
    params_dir = Column(String, default=None)
    knobs = Column(JSON, default=None)
    score = Column(Float, default=None)
    shared_param_id = Column(String, default=None)
    out_shared_param_id = Column(String, default=None)
    datetime_stopped = Column(DateTime, default=None)

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
    