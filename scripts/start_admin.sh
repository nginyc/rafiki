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

docker run --rm --name $ADMIN_HOST \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -e ADMIN_HOST=$ADMIN_HOST \
  -e ADMIN_PORT=$ADMIN_PORT \
  -e ADVISOR_HOST=$ADVISOR_HOST \
  -e ADVISOR_PORT=$ADVISOR_PORT \
  -e REDIS_HOST=$REDIS_HOST \
  -e REDIS_PORT=$REDIS_PORT \
  -e PREDICTOR_PORT=$PREDICTOR_PORT \
  -e RAFIKI_ADDR=$RAFIKI_ADDR \
  -e RAFIKI_IMAGE_WORKER=$RAFIKI_IMAGE_WORKER \
  -e RAFIKI_IMAGE_PREDICTOR=$RAFIKI_IMAGE_PREDICTOR \
  -e RAFIKI_VERSION=$RAFIKI_VERSION \
  -e LOGS_WORKDIR_PATH=$LOGS_WORKDIR_PATH \
  -e DATA_WORKDIR_PATH=$DATA_WORKDIR_PATH \
  -e LOGS_DOCKER_WORKDIR_PATH=$LOGS_DOCKER_WORKDIR_PATH \
  -e DATA_DOCKER_WORKDIR_PATH=$DATA_DOCKER_WORKDIR_PATH \
  -e DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $LOGS_WORKDIR_PATH:$LOGS_DOCKER_WORKDIR_PATH \
  -v $DATA_WORKDIR_PATH:$DATA_DOCKER_WORKDIR_PATH \
  -p $ADMIN_EXT_PORT:$ADMIN_PORT \
  $RAFIKI_IMAGE_ADMIN:$RAFIKI_VERSION