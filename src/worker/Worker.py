import logging
import abc
import os

from client import Client
from db import Database

class Worker(abc.ABC):
    def __init__(self, service_id, db=Database()):
        self._db = db
        self._service_id = service_id
        self._client = self._make_client()

    @abc.abstractmethod 
    def start(self):
        raise NotImplementedError()

    def _make_client(self):
        admin_host = os.environ['ADMIN_HOST']
        admin_port = os.environ['ADMIN_PORT']
        superadmin_email = os.environ['SUPERADMIN_EMAIL']
        superadmin_password = os.environ['SUPERADMIN_PASSWORD']
        client = Client(admin_host=admin_host, admin_port=admin_port)
        client.login(email=superadmin_email, password=superadmin_password)
        return client