from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

class TrainJobStatus():
    STARTED = 'STARTED'
    ERRORED = 'ERRORED'
    COMPLETED = 'COMPLETED'


class TrainJob(Base):
    __tablename__ = 'train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    budget_type = Column(String, nullable=False)
    budget_amount = Column(Integer, nullable=False)
    app_id = Column(String, ForeignKey('app.id'), nullable=False)
    status = Column(String, nullable=False, default=TrainJobStatus.STARTED)
    state_serialized = Column(Binary)
    datetime_completed = Column(DateTime, default=None)

