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

import abc
import sys
from typing import List, Dict
import os

class Dataset():
    def __init__(self, id: str, size_bytes: int):
        self.id = id
        self.size_bytes = size_bytes

class DataStore(abc.ABC):
    '''
        Persistent store for datasets.
    '''

    @abc.abstractmethod
    def save(self, data_file_path: str) -> Dataset:
        '''
            Persists a dataset in the local filesystem at file path, returning a ``Dataset`` abstraction containing a unique ID for the dataset.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, dataset_id: str) -> str:
        '''
            Loads a persisted dataset to the local filesystem, identified by ID, returning the file path to the dataset.
        '''
        raise NotImplementedError()

    @staticmethod
    def _get_size_bytes(data_file_path):
        st = os.stat(data_file_path)
        return st.st_size

    