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


LOG_FILE_PATH=$DOCKER_WORKDIR_PATH/$LOGS_DIR_PATH/test_integration.log

pytest -s -x test/test_users.py > $LOG_FILE_PATH
pytest -s -x test/test_models.py >> $LOG_FILE_PATH
pytest -s -x test/test_datasets.py >> $LOG_FILE_PATH
pytest -s -x test/test_train_jobs.py >> $LOG_FILE_PATH
pytest -s -x test/test_inference_jobs.py >> $LOG_FILE_PATH

pytest -s -x test/rafiki/utils/test_local_cache.py >> $LOG_FILE_PATH
pytest -s -x test/rafiki/advisor/test_make_advisor.py >> $LOG_FILE_PATH
pytest -s -x test/rafiki/kafka/test_kafka_cache.py >> $LOG_FILE_PATH
pytest -s -x test/rafiki/redis/test_param_cache.py >> $LOG_FILE_PATH

pytest -s -x test/test_workflow.py >> $LOG_FILE_PATH
