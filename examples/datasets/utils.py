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

import requests
import math
import tempfile
from tqdm import tqdm

def download_dataset_from_url(dataset_url):
    '''
        Download the dataset at URL over HTTP/HTTPS, ensuring that the dataset ends up in the local filesystem.
        Shows a progress bar.

        :param str dataset_url: Publicly available URL to download the dataset
        :returns: path of the dataset in the local filesystem (should be deleted after use) 
    '''
    print(f'Downloading dataset from {dataset_url}...')

    r = requests.get(dataset_url, stream=True)

    # Show a progress bar while downloading
    total_size = int(r.headers.get('content-length', 0)); 
    block_size = 1024
    iters = math.ceil(total_size / block_size) 
    with tempfile.NamedTemporaryFile(delete=False) as f:
        for data in tqdm(r.iter_content(block_size), total=iters, unit='KB'):
            f.write(data)

        return f.name