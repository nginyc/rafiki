import redis
import os

class Cache(object):

    def __init__(self,
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=os.environ.get('REDIS_PORt', '6379')):

        cache_connection_url = self._make_connection_url(
            host=host,
            port=port
        )

        self._connection_pool = redis.ConnectionPool.from_url(cache_connection_url)
        self._redis = redis.StrictRedis(connection_pool=self._connection_pool)

    def append_list(self, key, value):
        self._redis.rpush(key, value)
    
    def get_list_range(self, key, start, stop):
        return self._redis.lrange(key, start, stop)

    def get(self, key):
        return self._redis.get(key)

    def set(self, key, value):
        return self._redis.set(key, value)

    def delete(self, key):
        return self._redis.delete(key)

    def _make_connection_url(self, host, port):
        return 'redis://{}:{}'.format(host, port)