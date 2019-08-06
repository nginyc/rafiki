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

source ./scripts/kubernetes/.env.sh

python3 ./scripts/kubernetes/create_config.py \
$POSTGRES_PASSWORD \
$SUPERADMIN_PASSWORD \
$APP_SECRET \
$KUBERNETES_NETWORK \
$KUBERNETES_ADVERTISE_ADDR \
$RAFIKI_VERSION \
$RAFIKI_ADDR \
$ADMIN_EXT_PORT \
$WEB_ADMIN_EXT_PORT \
$POSTGRES_EXT_PORT \
$REDIS_EXT_PORT \
$ZOOKEEPER_EXT_PORT \
$KAFKA_EXT_PORT \
$HOST_WORKDIR_PATH \
$APP_MODE \
$POSTGRES_DUMP_FILE_PATH \
$DOCKER_NODE_LABEL_AVAILABLE_GPUS \
$DOCKER_NODE_LABEL_NUM_SERVICES \
$POSTGRES_USER \
$POSTGRES_DB \
$POSTGRES_HOST \
$POSTGRES_PORT \
$ADMIN_HOST \
$ADMIN_PORT \
$REDIS_HOST \
$REDIS_PORT \
$PREDICTOR_PORT \
$WEB_ADMIN_HOST \
$ZOOKEEPER_HOST \
$ZOOKEEPER_PORT \
$KAFKA_HOST \
$KAFKA_PORT \
$DOCKER_WORKDIR_PATH \
$DATA_DIR_PATH \
$LOGS_DIR_PATH \
$PARAMS_DIR_PATH \
$CONDA_ENVIORNMENT \
$WORKDIR_PATH \
$RAFIKI_IMAGE_ADMIN \
$RAFIKI_IMAGE_WEB_ADMIN \
$RAFIKI_IMAGE_WORKER \
$RAFIKI_IMAGE_PREDICTOR \
$IMAGE_POSTGRES \
$IMAGE_REDIS \
$IMAGE_ZOOKEEPER \
$IMAGE_KAFKA \
$PYTHONPATH \
$PYTHONUNBUFFERED \
$CONTAINER_MODE \
$CLUSTER_MODE
