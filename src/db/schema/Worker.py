from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

from common import WorkerStatus

class Worker(Base):
    __tablename__ = 'worker'

    id = Column(String, primary_key=True, default=generate_uuid)
    container_worker_id = Column(String, nullable=False)
    service_id = Column(String, ForeignKey('service.id'), nullable=False)
    status = Column(String, nullable=False, default=WorkerStatus.RUNNING)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
