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

from pprint import pprint
import os
import csv

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD

def seed_users(client, csv_file_path):
    with open(csv_file_path, 'rt', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lower() for name in reader.fieldnames]
        for row in reader:
            email = row['email']
            password = row['password']
            user_type = row['user_type']
            try:
                pprint(client.create_user(email, password, user_type))
            except Exception as e:
                print('Failed to create user `{}` due to:'.format(email))
                print(e)

if __name__ == '__main__':
    rafiki_host = os.environ.get('RAFIKI_HOST', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    web_admin_port = int(os.environ.get('WEB_ADMIN_EXT_PORT', 3001))
    user_email = os.environ.get('USER_EMAIL', SUPERADMIN_EMAIL)
    user_password = os.environ.get('USER_PASSWORD', SUPERADMIN_PASSWORD)
    csv_file_path = os.environ.get('CSV_FILE_PATH', 'examples/scripts/users.csv')

    # Initialize client
    client = Client(admin_host=rafiki_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)

    seed_users(client, csv_file_path)