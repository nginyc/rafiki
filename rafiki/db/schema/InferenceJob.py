from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

from rafiki.constants import InferenceJobStatus

class InferenceJob(Base):
    __tablename__ = 'inference_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    app = Column(String, nullable=False)
    app_version = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default=InferenceJobStatus.STARTED)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    query_service_id = Column(String, ForeignKey('service.id'))
    datetime_stopped = Column(DateTime, default=None)
