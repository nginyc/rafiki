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

LOG_FILE_PATH=$PWD/logs/start_web_admin.log

source ./scripts/utils.sh

title "Starting Rafiki's Web Admin..."
(docker run --rm --name $WEB_ADMIN_HOST \
  --network $DOCKER_NETWORK \
  -e RAFIKI_ADDR=$RAFIKI_ADDR \
  -e ADMIN_EXT_PORT=$ADMIN_EXT_PORT \
  -p $WEB_ADMIN_EXT_PORT:3001 \
  $RAFIKI_IMAGE_WEB_ADMIN:$RAFIKI_VERSION \
  &> $LOG_FILE_PATH) &

ensure_stable "Rafiki's Web Admin" $LOG_FILE_PATH 10
