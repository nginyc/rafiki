from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSON

from .utils import generate_uuid
from .Base import Base

class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(String, primary_key=True, default=generate_uuid)
    dataset_type = Column(String, nullable=False)
    params = Column(JSON, nullable=False)

