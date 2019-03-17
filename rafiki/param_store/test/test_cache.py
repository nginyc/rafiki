import pytest

from rafiki.test.utils import global_setup
from rafiki.param_store.cache import Cache
import numpy as np

class TestCache():
    @pytest.fixture(scope='class', autouse=True)
    def params(self):
        return {
            '1': np.array([[1., 2.], [3., 4.]]),
            '2': np.array([[2., 2.], [3., 4.]]),
            '3': np.array([[3., 2.], [3., 4.]]),
            '4': np.array([[4., 2.], [3., 4.]]),
            '5': np.array([[5., 2.], [3., 4.]])
        }

    def test_store_value(self, params):
        cache = Cache(size=4)
        cache.put('1', params['1'])
        assert cache.get('1') is params['1']

    def test_replace_value(self, params):
        cache = Cache(size=4)
        cache.put('1', params['1'])
        cache.put('1', params['2'])
        assert cache.get('1') is params['2']

    def test_get_missing_value(self, params):
        cache = Cache(size=4)
        cache.put('1', params['1'])
        assert cache.get('2') is None

    def test_evict_value(self, params):
        cache = Cache(size=4)
        cache.put('1', params['1'])
        cache.put('2', params['2'])
        cache.put('3', params['3'])
        cache.put('4', params['4'])
        assert len(cache) == 4
        cache.put('5', params['5'])
        assert len(cache) == 4
        assert cache.get('1') is None

    def test_get_and_put_values(self, params):
        cache = Cache(size=4)
        assert cache.get('1') is None
        cache.put('1', params['1'])
        cache.put('2', params['2'])
        assert cache.get('1') is params['1']
        assert cache.get('2') is params['2']
        assert cache.get('3') is None
        cache.put('3', params['3'])
        cache.put('4', params['4'])
        cache.put('5', params['5'])
        assert cache.get('4') is params['4']
        assert cache.get('5') is params['5']
    
    def test_repeatedly_evict_values(self, params):
        cache = Cache(size=2)
        cache.put('1', params['1'])
        cache.put('2', params['2'])
        cache.put('3', params['3'])
        cache.put('2', params['2'])
        assert cache.get('2') is params['2']
        cache.put('1', params['1'])
        cache.put('4', params['4'])
        assert cache.get('4') is params['4']
        cache.put('1', params['1'])
        cache.put('2', params['2'])
        cache.put('3', params['3'])
        assert cache.get('3') is params['3']
    
    def test_evict_least_recently_used_value(self, params):
        cache = Cache(size=4)
        cache.put('1', params['1'])
        cache.put('2', params['2'])
        cache.put('3', params['3'])
        cache.put('4', params['4'])
        cache.put('1', params['1']) # Put 1 again
        assert cache.get('2') is params['2'] # Get 2 again 
        cache.put('5', params['5'])
        assert cache.get('3') is None # 3 is least recently used