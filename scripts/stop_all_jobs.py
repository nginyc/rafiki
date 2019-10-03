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

import os

from singaauto.client import Client
from singaauto.config import SUPERADMIN_EMAIL

if __name__ == '__main__':
    singaauto_host = os.environ.get('SINGAAUTO_HOST', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    user_email = SUPERADMIN_EMAIL
    user_password = os.environ.get('SUPERADMIN_PASSWORD', 'singaauto')

    # Initialize client
    client = Client(admin_host=singaauto_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)
    print(client.stop_all_jobs())