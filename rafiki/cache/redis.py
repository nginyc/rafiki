import uuid
import time
import logging
import msgpack

logger = logging.getLogger(__name__)

REDIS_LOCK_EXPIRE_SECONDS = 60
REDIS_LOCK_WAIT_SLEEP_SECONDS = 0.1

class RedisSession(object):
    '''
        Wraps Redis.

        If host & port are None, underlying Redis connection will be mocked.

        Namespace usage:
            
            <namespace>:lock  | 0-1 lock for this namespace, 1 for acquired    
    '''
    def __init__(self,
                namespace, # Namespace to use for redis
                redis_host=None,
                redis_port=None):
        self._uid = str(uuid.uuid4()) # Process identifier for distributed locking
        self._namespace = namespace
        self._redis = self._make_redis_client(redis_host, redis_port)

    def acquire_lock(self):
        lock_value = self._uid        
        lock_name = self._get_redis_name('lock')

        # Keep trying to acquire lock
        res = None
        while not res:        
            res = self._redis.set(lock_name, lock_value, nx=True, ex=REDIS_LOCK_EXPIRE_SECONDS)
            if not res:
                sleep_secs = REDIS_LOCK_WAIT_SLEEP_SECONDS 
                logger.info('Waiting for lock to be released, sleeping for {}s...'.format(sleep_secs))
                time.sleep(sleep_secs)

    def release_lock(self):
        lock_value = self._uid     
        lock_name = self._get_redis_name('lock')   

        # Only release lock if it's confirmed to be the one I acquired
        # Possible that it was a lock acquired by someone else after my lock expired
        cur_lock_value = self._redis.get(lock_name)
        cur_lock_value = cur_lock_value.decode() if cur_lock_value is not None else None
        if cur_lock_value == lock_value: 
            self._redis.delete(lock_name)
        else:
            logger.info('Lock is not mine - not releasing...')

    def get(self, name):
        key = self._get_redis_name(name)
        value = self._redis.get(key)
        value = self._parse_value(value)
        return value

    def set(self, name, value):
        key = self._get_redis_name(name)
        value = self._encode_value(value)
        self._redis.set(key, value)

    def delete(self, *names):
        keys = [self._get_redis_name(x) for x in names]
        self._redis.delete(*keys)

    def delete_pattern(self, pattern):
        key_patt = self._get_redis_name(pattern)
        keys = self._redis.keys(key_patt)
        if len(keys) > 0:
            self._redis.delete(*keys)

    def _encode_value(self, value):
        value = msgpack.packb(value, use_bin_type=True)
        return value

    def _parse_value(self, value):
        if value is None:
            return value
        value = msgpack.unpackb(value, raw=False)
        return value
    
    def _get_redis_name(self, name):
        return '{}:{}'.format(self._namespace, name)

    def _make_redis_client(self, host, port):
        if host is not None and port is not None:
            import redis
            cache_connection_url = 'redis://{}:{}'.format(host, port)
            connection_pool = redis.ConnectionPool.from_url(cache_connection_url)
            client = redis.StrictRedis(connection_pool=connection_pool, decode_responses=True)
            logger.info(f'Connecting to Redis at namespace {self._namespace}...')
        else:
            client = MockRedis()
            logger.info('Using mock Redis...')

        return client

class MockRedis():
    data = {}
    
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
        if len(keys) == 0:
            raise ValueError('Need at least 1 key to delete')

        for key in keys:
            del self.data[key]
