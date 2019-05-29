import pytest
from unittest.mock import MagicMock
import numpy
import random as rand

@pytest.fixture(scope='session', autouse=True)
def global_setup():
    '''
    Does global setup for tests (e.g. seeding RNG)
    '''
    rand.seed(0)
    numpy.random.seed(0)

@pytest.fixture(scope='session', autouse=True)
def mock_redis():
    import redis
    redis.ConnectionPool.from_url = MagicMock() 
    redis.StrictRedis = RedisMock

class RedisMock():
    data = {}
    
    def __init__(self, **kwargs):
        pass

    def get(self, key):
        value = self.data.get(key)
        if isinstance(value, str):
            return value.encode()
        
        return value
    
    def set(self, key, value, **kwargs):
        is_set = (key in self.data)
        self.data[key] = value
        return not is_set

    def keys(self, patt):
        # TODO: Do more accurate implementation based on pattern
        return self.data.keys()
    
    def delete(self, *keys):
        for key in keys:
            del self.data[key]

