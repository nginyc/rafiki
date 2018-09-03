import numpy as np
from db import Database, DatabaseConfig

from model import unserialize_model, serialize_model
from .auth import hash_password, if_hash_matches_password

class WorkerManager(object):
    def __init__(self):
        pass

    def revise_workers(self, train_jobs):
        pass
    