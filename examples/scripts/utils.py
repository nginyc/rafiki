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

import string
import random
import time

from rafiki.constants import TrainJobStatus

# Generates a random ID
def gen_id(length=16):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

# Blocks until a train job has stopped
def wait_until_train_job_has_stopped(client, app, timeout=60*20, tick=10):
    length = 0
    while True:
        train_job = client.get_train_job(app)
        status = train_job['status']

        if status == TrainJobStatus.ERRORED:
            raise Exception('Train job has errored.')
        elif status == TrainJobStatus.STOPPED:
            # Unblock
            return

        # Still running...
        if timeout is not None and length >= timeout:
            raise TimeoutError('Train job is running for too long')

        length += tick
        time.sleep(tick)