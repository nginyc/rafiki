from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

class TrainJobWorker(Base):
    __tablename__ = 'train_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    model_id = Column(String, ForeignKey('model.id'), nullable=False)