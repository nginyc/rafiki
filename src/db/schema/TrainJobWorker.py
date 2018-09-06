from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

from common import TrainJobWorkerStatus

class TrainJobWorker(Base):
    __tablename__ = 'train_job_worker'

    id = Column(String, primary_key=True, default=generate_uuid)
    train_job_id = Column(String, ForeignKey('train_job.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    service_id = Column(String)
    status = Column(String, nullable=False, default=TrainJobWorkerStatus.STARTED)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    replicas = Column(Integer, nullable=False, default=0)

