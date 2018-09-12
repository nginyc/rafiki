from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

from rafiki.constants import TrainJobStatus

class TrainJob(Base):
    __tablename__ = 'train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    app = Column(String, nullable=False)
    app_version = Column(Integer, nullable=False)
    task = Column(String, nullable=False)
    train_dataset_uri = Column(String, nullable=False)
    test_dataset_uri = Column(String, nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    budget_type = Column(String, nullable=False)
    budget_amount = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default=TrainJobStatus.STARTED)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    datetime_completed = Column(DateTime, default=None)
