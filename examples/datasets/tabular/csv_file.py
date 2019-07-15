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

import pandas as pd
import json
import tempfile
import shutil
import os

from pathlib import Path
from sklearn.model_selection import train_test_split

def load(dataset_url, out_train_dataset_path, out_val_dataset_path):
    '''
        Splits a standard CSV file into train & validation datasets, as per the DatasetType `TABULAR`.

        :param str dataset_url: URL to download the dataset CSV file
        :param str out_train_dataset_path: Path to save the output train dataset (CSV) file
        :param str out_val_dataset_path: Path to save the output test dataset (CSV) file
    '''

    print('Loading & splitting dataset...')
    X = pd.read_csv(dataset_url)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)

    _write_dataset(X_train, out_train_dataset_path)
    print('Train dataset file is saved at {}'.format(out_train_dataset_path))

    _write_dataset(X_test, out_val_dataset_path)
    print('Validation dataset file is saved at {}'.format(out_val_dataset_path))

def _write_dataset(data, out_dataset_path):
    data.to_csv(out_dataset_path, index=False)
