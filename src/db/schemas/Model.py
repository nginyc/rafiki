from sqlalchemy import Column, String, ForeignKey, Integer, Float, Binary, DateTime
from sqlalchemy.dialects.postgresql import JSON

from .utils import generate_uuid, generate_datetime
from .Base import Base

class Model(Base):
    __tablename__ = 'model'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)
    name = Column(String, unique=True, nullable=False)
    task = Column(String, nullable=False)
    model_serialized = Column(Binary, nullable=False)
    