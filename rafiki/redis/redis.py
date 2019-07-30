#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

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
        value = self._decode_value(value)
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
    
    def add_to_set(self, name, *values):
        key = self._get_redis_name(name)
        values = [self._encode_value(x) for x in values]
        self._redis.sadd(key, *values)

    def delete_from_set(self, name, value):
        key = self._get_redis_name(name)
        value = self._encode_value(value)
        self._redis.srem(key, value)
    
    def list_set(self, name):
        key = self._get_redis_name(name)
        values = self._redis.smembers(key)
        return [self._decode_value(x) for x in values]

    def prepend_to_list(self, name, *values):
        key = self._get_redis_name(name)
        values = [self._encode_value(x) for x in values]
        self._redis.lpush(key, *values)
    
    def pop_from_list(self, name):
        key = self._get_redis_name(name)
        value = self._redis.rpop(key)
        value = self._decode_value(value)
        return value

    def _encode_value(self, value):
        value = msgpack.packb(value, use_bin_type=True)
        return value

    def _decode_value(self, value):
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
        return value
    
    def set(self, key, value, **kwargs):
        is_set = (key in self.data)
        if isinstance(value, str):
            value = value.encode()

        self.data[key] = value
        return not is_set

    def keys(self, patt):
        # TODO: Do more accurate implementation based on pattern
        return self.data.keys()

    def sadd(self, key, *values):
        if key not in self.data:
            self.data[key] = set()

        if not isinstance(self.data[key], set):
            raise KeyError(f'Value at key "{key}" is not a set')

        for value in values:
            if isinstance(value, str):
                value = value.encode()
            self.data[key].add(value)
    
    def srem(self, key, *values):
        if key not in self.data:
            self.data[key] = set()

        if not isinstance(self.data[key], set):
            raise KeyError(f'Value at key "{key}" is not a set')
        
        for value in values:
            if isinstance(value, str):
                value = value.encode()
            self.data[key].remove(value)

    def smembers(self, key):
        if key not in self.data:
            return []

        if not isinstance(self.data[key], set):
            raise KeyError(f'Value at key "{key}" is not a set')

        return list(self.data[key])

    def lpush(self, key, *values):
        if key not in self.data:
            self.data[key] = list()

        if not isinstance(self.data[key], list):
            raise KeyError(f'Value at key "{key}" is not a list')

        for value in values:
            if isinstance(value, str):
                value = value.encode()
            self.data[key].insert(0, value)
    
    def rpop(self, key):
        if key not in self.data:
            self.data[key] = list()

        if not isinstance(self.data[key], list):
            raise KeyError(f'Value at key "{key}" is not a list')
        
        if len(self.data[key]) == 0:
            return None

        return self.data[key].pop()

    def delete(self, *keys):
        if len(keys) == 0:
            raise ValueError('Need at least 1 key to delete')

        for key in keys:
            del self.data[key]
