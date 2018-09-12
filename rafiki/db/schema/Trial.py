import datetime
from sqlalchemy import Column, String, ForeignKey, Integer, Float, DateTime
from sqlalchemy.dialects.postgresql import JSON

from rafiki.constants import TrialStatus

from .utils import generate_uuid, generate_datetime
from .Base import Base

class Trial(Base):
    __tablename__ = 'trial'

    id = Column(String, primary_key=True, default=generate_uuid)
    hyperparameters = Column(JSON, nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    train_job_id = Column(String, ForeignKey('train_job.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    status = Column(String, nullable=False, default=TrialStatus.STARTED)
    score = Column(Float, default=0)
    parameters = Column(JSON, default=None)
    datetime_completed = Column(DateTime, default=None)
    