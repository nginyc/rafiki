from sqlalchemy import Column, String, ForeignKey, DateTime, Binary

from .utils import generate_uuid
from .Base import Base

class User(Base):
    __tablename__ = 'user'

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(Binary, nullable=False)
    user_type = Column(String, nullable=False)
    