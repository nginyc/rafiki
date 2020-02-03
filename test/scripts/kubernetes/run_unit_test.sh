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


LOG_FILE_PATH=$DOCKER_WORKDIR_PATH/$LOGS_DIR_PATH/test_unit.log

pytest -s -x test/unit/test_metastore_class.py > $LOG_FILE_PATH
pytest -s -x test/unit/test_admin_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_services_manager_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_advisor_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_container_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_data_store_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_kafka_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_model_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_param_store_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_predictor_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_redis_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_worker_class.py >> $LOG_FILE_PATH
pytest -s -x test/unit/test_utils_class.py >> $LOG_FILE_PATH
