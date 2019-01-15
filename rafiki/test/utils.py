import pytest
import numpy
import random as rand

@pytest.fixture(scope='session', autouse=True)
def global_setup():
    '''
    Does global setup for tests (e.g. seeding RNG)
    '''
    rand.seed(0)
    numpy.random.seed(0)
