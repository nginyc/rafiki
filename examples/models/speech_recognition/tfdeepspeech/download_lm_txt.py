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

import gzip
import io
import sys
import time
import os
from urllib import request
import argparse

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download_lm_txt(lm_txt_path):
    url = 'http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz'
    data_upper = '/tmp/upper.txt.gz'
    request.urlretrieve(url, data_upper, reporthook)

    # Convert to lowercase and cleanup.
    with open(lm_txt_path, 'w', encoding='utf-8') as lower:
        with io.TextIOWrapper(io.BufferedReader(gzip.open(data_upper)), encoding='utf8') as upper:
            for line in upper:
                lower.write(line.lower())

    print(f'Converted to {lm_txt_path}')
    print('Starting cleanup')
    os.remove(data_upper)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=int, default='/tmp/lower.txt')
    args = parser.parse_args()

    # Download LM text
    download_lm_txt(args.path)
