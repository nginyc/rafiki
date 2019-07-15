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

import pytest

from rafiki.constants import UserType
from test.utils import global_setup, gen, gen_email, make_admin, make_app_dev, make_model_dev

class TestUsers():
    
    def test_admin_create_users(self):
        admin = make_admin()
        model_dev_email = gen_email()
        app_dev_email = gen_email()
        password = gen()

        # Create model dev
        model_dev = admin.create_user(model_dev_email, password, UserType.MODEL_DEVELOPER)
        assert 'id' in model_dev
        model_dev_id = model_dev['id']

        # Create app dev
        app_dev = admin.create_user(app_dev_email, password, UserType.APP_DEVELOPER)
        assert 'id' in app_dev
        app_dev_id = model_dev['id']

        # Created users are in list of users
        users = admin.get_users()
        assert any([x['id'] == model_dev_id for x in users])
        assert any([x['id'] == app_dev_id for x in users])
    
    
    def test_admin_ban_users(self):
        admin = make_admin()
        model_dev_email = gen_email()
        app_dev_email = gen_email()
        password = gen()
        model_dev = make_model_dev(email=model_dev_email, password=password)
        app_dev = make_app_dev(email=app_dev_email, password=password)

        # Ban both users
        admin.ban_user(model_dev_email)
        admin.ban_user(app_dev_email)

        # Both users cannot login again
        with pytest.raises(Exception):
            model_dev.login(model_dev_email, password)
        
        with pytest.raises(Exception):
            app_dev.login(app_dev_email, password)
        

    def test_model_dev_cant_manage_users(self):
        ban_email = gen_email()
        email = gen_email()
        password = gen()
        model_dev = make_model_dev()
        app_dev = make_app_dev(email=ban_email)

        # Can't create user
        with pytest.raises(Exception):
            model_dev.create_user(email, password, UserType.ADMIN)

        # Can't list users
        with pytest.raises(Exception):
            model_dev.get_users()

        # Can't ban user
        with pytest.raises(Exception):
            model_dev.ban_user(ban_email)
        

    def test_app_dev_cant_manage_users(self):
        ban_email = f'{gen()}@rafiki'
        email = gen_email()
        password = gen()
        model_dev = make_model_dev(email=ban_email)
        app_dev = make_app_dev()

        # Can't create user
        with pytest.raises(Exception):
            app_dev.create_user(email, password, UserType.ADMIN)

        # Can't list users
        with pytest.raises(Exception):
            app_dev.get_users()

        # Can't ban user
        with pytest.raises(Exception):
            app_dev.ban_user(ban_email)
