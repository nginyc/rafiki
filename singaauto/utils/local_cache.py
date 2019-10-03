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

class _CacheNode():
    def __init__(self):
        self.key = None
        self.val = None
        self.prev = None
        self.next = None

class LocalCache():
    def __init__(self, size: int):
        assert size > 0
        self._size = size
        self._head = None # Doubly linked list of nodes
        self._tail = None # Tail of linked list
        self._key_to_node = {} # { <key>: <node> }

    def __len__(self):
        return len(self._key_to_node)

    def __contains__(self, item):
        return item in self._key_to_node

    @property
    def size(self) -> int:
        return self._size

    def put(self, key: str, value):
        # If node exists, update value and move it to front
        if key in self._key_to_node:
            node = self._key_to_node[key]
            node.val = value
            self._move_to_front(node)
            return
            
        # Get node to place data
        node = self._maybe_evict()

        # Populate data 
        node.key = key
        node.val = value
        self._key_to_node[key] = node
        self._insert_to_front(node)

    def get(self, key: str):
        if key in self._key_to_node:
            node = self._key_to_node[key]
            self._move_to_front(node)
            return node.val
        else:
            return None

    def _maybe_evict(self):
        # If there's space
        if len(self._key_to_node) < self._size:
            return _CacheNode()
        
        # Evict tail
        node = self._tail
        if node.prev is not None:
            node.prev.next = None
        self._tail = node.prev
        del self._key_to_node[node.key]
        return node

    def _insert_to_front(self, node):
        # If list is empty
        if self._head is None:
            self._head = node
            self._tail = node
            return

        node.next = self._head
        node.prev = None
        self._head.prev = node
        self._head = node

    def _move_to_front(self, node):
        # If node is already in front
        if node is self._head:
            return

        # If node is tail, update tail to prev node
        if node is self._tail:
            self._tail = node.prev
        
        # Remove node from list
        node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev

        # Move to head
        node.next = self._head
        self._head.prev = node
        node.prev = None
        self._head = node

    def __str__(self):
        return 'Param cache has {} / {} items'.format(len(self), self.size)

