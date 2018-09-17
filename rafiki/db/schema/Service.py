from sqlalchemy import Column, String, ForeignKey, Integer, Binary, DateTime

from .utils import generate_uuid, generate_datetime
from .Base import Base

from rafiki.constants import ServiceStatus

class Service(Base):
    __tablename__ = 'service'

    id = Column(String, primary_key=True, default=generate_uuid)
    service_type = Column(String, nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_stopped = Column(DateTime, default=None)
    status = Column(String, nullable=False, default=ServiceStatus.RUNNING)
    docker_image = Column(String, nullable=False)
    container_manager_type = Column(String, nullable=False)
    replicas = Column(Integer, default=0)
    ext_hostname = Column(String)
    ext_port = Column(Integer)
    hostname = Column(String)
    port = Column(Integer)
    container_service_name = Column(String)
    container_service_id = Column(String)

