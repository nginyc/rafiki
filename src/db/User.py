from sqlalchemy import Column, String, ForeignKey, DateTime

from .utils import generate_uuid
from .Base import Base

class UserType():
    ADMIN = 'ADMIN'
    MODEL_DEVELOPER = 'MODEL_DEVELOPER',
    APP_DEVELOPER = 'APP_DEVELOPER',
    USER = 'USER'


class User(Base):
    __tablename__ = 'user'

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    user_type = Column(String, nullable=False)
    