from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

class InferenceJobWorker(Base):
    __tablename__ = 'inference_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    inference_job_id = Column(String, ForeignKey('inference_job.id'))
    trial_id = Column(String, ForeignKey('trial.id'), nullable=False)