from sqlalchemy import Column, String, ForeignKey, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

class App(Base):
    __tablename__ = 'app'

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)
    task = Column(String, nullable=False)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)
    train_dataset_uri = Column(String, nullable=False)
    test_dataset_uri = Column(String, nullable=False)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)