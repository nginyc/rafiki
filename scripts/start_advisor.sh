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

LOG_FILE_PATH=$PWD/logs/start_advisor.log

# Mount whole project folder with code for dev for shorter iterations
if [ $APP_MODE = "DEV" ]; then
  VOLUME_MOUNTS="-v $PWD:$DOCKER_WORKDIR_PATH"
else
  VOLUME_MOUNTS="-v $HOST_WORKDIR_PATH/$LOGS_DIR_PATH:$DOCKER_WORKDIR_PATH/$LOGS_DIR_PATH"
fi

source ./scripts/utils.sh

title "Starting Rafiki's Advisor..."
(docker run --rm --name $ADVISOR_HOST \
  --network $DOCKER_NETWORK \
  -e WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
  -e LOGS_DIR_PATH=$LOGS_DIR_PATH \
  $VOLUME_MOUNTS \
  -p $ADVISOR_EXT_PORT:$ADVISOR_PORT \
  $RAFIKI_IMAGE_ADVISOR:$RAFIKI_VERSION \
  &> $LOG_FILE_PATH) &

ensure_stable "Rafiki's Advisor" $LOG_FILE_PATH 10