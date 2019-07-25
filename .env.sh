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

# Core secrets for Rafiki - change these in production!
export POSTGRES_PASSWORD=rafiki
export SUPERADMIN_PASSWORD=rafiki
export APP_SECRET=rafiki

# Core external configuration for Rafiki
export DOCKER_NETWORK=rafiki
export DOCKER_SWARM_ADVERTISE_ADDR=127.0.0.1
export RAFIKI_VERSION=0.2.0
export RAFIKI_ADDR=127.0.0.1
export ADMIN_EXT_PORT=3000
export WEB_ADMIN_EXT_PORT=3001
export POSTGRES_EXT_PORT=5433
export REDIS_EXT_PORT=6380
export ZOOKEEPER_EXT_PORT=2181
export KAFKA_EXT_PORT=9092
export HOST_WORKDIR_PATH=$PWD
export APP_MODE=DEV # DEV or PROD
export POSTGRES_DUMP_FILE_PATH=$PWD/db_dump.sql # PostgreSQL database dump file
export DOCKER_NODE_LABEL_AVAILABLE_GPUS=available_gpus # Docker node label for no. of services currently running on the node
export DOCKER_NODE_LABEL_NUM_SERVICES=num_services # Docker node label for no. of services currently running on the node

# Internal credentials for Rafiki's components
export POSTGRES_USER=rafiki
export POSTGRES_DB=rafiki

# Internal hosts & ports and configuration for Rafiki's components 
export POSTGRES_HOST=rafiki_db
export POSTGRES_PORT=5432
export ADMIN_HOST=rafiki_admin
export ADMIN_PORT=3000
export REDIS_HOST=rafiki_redis
export REDIS_PORT=6379
export PREDICTOR_PORT=3003
export WEB_ADMIN_HOST=rafiki_admin_web
export ZOOKEEPER_HOST=rafiki_zookeeper
export ZOOKEEPER_PORT=2181
export KAFKA_HOST=rafiki_kafka
export KAFKA_PORT=9092
export DOCKER_WORKDIR_PATH=/root
export DATA_DIR_PATH=data # Shares a data folder with containers, relative to workdir
export LOGS_DIR_PATH=logs # Shares a folder with containers that stores components' logs, relative to workdir
export PARAMS_DIR_PATH=params # Shares a folder with containers that stores model parameters, relative to workdir
export CONDA_ENVIORNMENT=rafiki
export WORKDIR_PATH=$HOST_WORKDIR_PATH # Specifying workdir if Python programs are run natively

# Docker images for Rafiki's custom components
export RAFIKI_IMAGE_ADMIN=rafikiai/rafiki_admin
export RAFIKI_IMAGE_WEB_ADMIN=rafikiai/rafiki_admin_web
export RAFIKI_IMAGE_WORKER=rafikiai/rafiki_worker
export RAFIKI_IMAGE_PREDICTOR=rafikiai/rafiki_predictor

# Docker images for dependent services
export IMAGE_POSTGRES=postgres:10.5-alpine
export IMAGE_REDIS=redis:5.0.3-alpine3.8
export IMAGE_ZOOKEEPER=zookeeper:3.5
export IMAGE_KAFKA=wurstmeister/kafka:2.12-2.1.1

# Utility configuration
export PYTHONPATH=$PWD # Ensures that `rafiki` module can be imported at project root
export PYTHONUNBUFFERED=1 # Ensures logs from Python appear instantly 
