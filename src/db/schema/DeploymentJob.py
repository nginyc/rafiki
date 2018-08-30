from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

from common import DeployStatus

class DeploymentJob(Base):
    __tablename__ = 'deployment_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    app_id = Column(String, ForeignKey('app.id'), nullable=False)
    status = Column(String, nullable=False, default=DeployStatus.STARTED)
    datetime_stopped = Column(DateTime, default=None)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)

