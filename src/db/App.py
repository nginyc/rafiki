from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship

from .utils import generate_uuid
from .Base import Base

class App(Base):
    __tablename__ = 'app'

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)
    task = Column(String, nullable=False)
    dataset_id = Column(String, ForeignKey('dataset.id'), nullable=False)
    dataset = relationship('Dataset', uselist=False)
    