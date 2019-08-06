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

import json
import sys
import os

if __name__ == '__main__':
    POSTGRES_PASSWORD = sys.argv[1]
    SUPERADMIN_PASSWORD = sys.argv[2]
    APP_SECRET = sys.argv[3]
    KUBERNETES_NETWORK = sys.argv[4]
    KUBERNETES_ADVERTISE_ADDR = sys.argv[5]
    RAFIKI_VERSION = sys.argv[6]
    RAFIKI_ADDR = sys.argv[7]
    ADMIN_EXT_PORT = sys.argv[8]
    WEB_ADMIN_EXT_PORT = sys.argv[9]
    POSTGRES_EXT_PORT = sys.argv[10]
    REDIS_EXT_PORT = sys.argv[11]
    ZOOKEEPER_EXT_PORT = sys.argv[12]
    KAFKA_EXT_PORT = sys.argv[13]
    HOST_WORKDIR_PATH = sys.argv[14]
    APP_MODE = sys.argv[15] 
    POSTGRES_DUMP_FILE_PATH = sys.argv[16] 
    DOCKER_NODE_LABEL_AVAILABLE_GPUS = sys.argv[17] 
    DOCKER_NODE_LABEL_NUM_SERVICES = sys.argv[18]

    POSTGRES_USER = sys.argv[19]
    POSTGRES_DB = sys.argv[20]
 
    POSTGRES_HOST = sys.argv[21]
    POSTGRES_PORT = sys.argv[22]
    ADMIN_HOST = sys.argv[23]
    ADMIN_PORT = sys.argv[24]
    REDIS_HOST = sys.argv[25]
    REDIS_PORT = sys.argv[26]
    PREDICTOR_PORT = sys.argv[27]
    WEB_ADMIN_HOST = sys.argv[28]
    ZOOKEEPER_HOST = sys.argv[29]
    ZOOKEEPER_PORT = sys.argv[30]
    KAFKA_HOST = sys.argv[31]
    KAFKA_PORT = sys.argv[32]
    DOCKER_WORKDIR_PATH = sys.argv[33]
    DATA_DIR_PATH = sys.argv[34] 
    LOGS_DIR_PATH = sys.argv[35] 
    PARAMS_DIR_PATH = sys.argv[36] 
    CONDA_ENVIORNMENT = sys.argv[37]
    WORKDIR_PATH = sys.argv[38] 

    RAFIKI_IMAGE_ADMIN = sys.argv[39]
    RAFIKI_IMAGE_WEB_ADMIN = sys.argv[40]
    RAFIKI_IMAGE_WORKER = sys.argv[41]
    RAFIKI_IMAGE_PREDICTOR = sys.argv[42]

    IMAGE_POSTGRES = sys.argv[43]
    IMAGE_REDIS = sys.argv[44]
    IMAGE_ZOOKEEPER = sys.argv[45]
    IMAGE_KAFKA = sys.argv[46]

    PYTHONPATH = sys.argv[47] 
    PYTHONUNBUFFERED = sys.argv[48]
    CONTAINER_MODE = sys.argv[49]
    CLUSTER_MODE = sys.argv[50]

    #zk service
    content = {}
    content.setdefault('apiVersion', 'v1')
    content.setdefault('kind', 'Service')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', ZOOKEEPER_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', ZOOKEEPER_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('type', 'NodePort')
    ports = spec.setdefault('ports', [])
    ports.append({'port': int(ZOOKEEPER_PORT), 'targetPort': int(ZOOKEEPER_PORT), 'nodePort': int(ZOOKEEPER_EXT_PORT)})
    spec.setdefault('selector', {'name': ZOOKEEPER_HOST})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_zookeeper_service.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #zk deployment
    content = {}
    content.setdefault('apiVersion', 'apps/v1')      
    content.setdefault('kind', 'Deployment')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', ZOOKEEPER_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', ZOOKEEPER_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('replicas', 1)
    spec.setdefault('selector', {'matchLabels': {'name': ZOOKEEPER_HOST}})
    template = spec.setdefault('template', {})
    template.setdefault('metadata', {'labels': {'name': ZOOKEEPER_HOST}})
    container = {}
    container.setdefault('name', ZOOKEEPER_HOST)
    container.setdefault('image', IMAGE_ZOOKEEPER)
    env = []
    env.append({'name': 'CONTAINER_MODE', 'value': CONTAINER_MODE})
    container.setdefault('env', env)
    template.setdefault('spec', {'containers': [container]})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_zookeeper_deployment.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #kafka service
    content = {}
    content.setdefault('apiVersion', 'v1')
    content.setdefault('kind', 'Service')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', KAFKA_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', KAFKA_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('type', 'NodePort')
    ports = spec.setdefault('ports', [])
    ports.append({'port': int(KAFKA_PORT), 'targetPort': int(KAFKA_PORT), 'nodePort': int(KAFKA_EXT_PORT)})
    spec.setdefault('selector', {'name': KAFKA_HOST})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_kafka_service.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #kafka deployment
    content = {}
    content.setdefault('apiVersion', 'apps/v1')      
    content.setdefault('kind', 'Deployment')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', KAFKA_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', KAFKA_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('replicas', 1)
    spec.setdefault('selector', {'matchLabels': {'name': KAFKA_HOST}})
    template = spec.setdefault('template', {})
    template.setdefault('metadata', {'labels': {'name': KAFKA_HOST}})
    container = {}
    container.setdefault('name', KAFKA_HOST)
    container.setdefault('image', IMAGE_KAFKA)
    env = []
    env.append({'name': 'CONTAINER_MODE', 'value': CONTAINER_MODE})
    env.append({'name': 'KAFKA_ZOOKEEPER_CONNECT', 'value': f'{ZOOKEEPER_HOST}:{ZOOKEEPER_PORT}'})
    env.append({'name': 'KAFKA_ADVERTISED_HOST_NAME', 'value': KAFKA_HOST})
    env.append({'name': 'KAFKA_ADVERTISED_PORT', 'value': KAFKA_PORT})
    container.setdefault('env', env)
    template.setdefault('spec', {'containers': [container]})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_kafka_deployment.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #db service
    if CLUSTER_MODE == "SINGLE":
        content = {}
        content.setdefault('apiVersion', 'v1')
        content.setdefault('kind', 'Service')
        metadata = content.setdefault('metadata', {})
        metadata.setdefault('name', POSTGRES_HOST)
        labels = metadata.setdefault('labels', {})
        labels.setdefault('name', POSTGRES_HOST)
        spec = content.setdefault('spec', {})
        spec.setdefault('type', 'NodePort')
        ports = spec.setdefault('ports', [])
        ports.append({'port': int(POSTGRES_PORT), 'targetPort': int(POSTGRES_PORT), 'nodePort': int(POSTGRES_EXT_PORT)})
        spec.setdefault('selector', {'name': POSTGRES_HOST})
        with open(f'{PYTHONPATH}/scripts/kubernetes/start_db_service.json', 'w') as f:
            f.write(json.dumps(content, indent=4))

        #db deployment
        content = {}
        content.setdefault('apiVersion', 'apps/v1')
        content.setdefault('kind', 'Deployment')
        metadata = content.setdefault('metadata', {})
        metadata.setdefault('name', POSTGRES_HOST)
        labels = metadata.setdefault('labels', {})
        labels.setdefault('name', POSTGRES_HOST)
        spec = content.setdefault('spec', {})
        spec.setdefault('replicas', 1)
        spec.setdefault('selector', {'matchLabels': {'name': POSTGRES_HOST}})
        template = spec.setdefault('template', {})
        template.setdefault('metadata', {'labels': {'name': POSTGRES_HOST}})
        container = {}
        container.setdefault('name', POSTGRES_HOST)
        container.setdefault('image', IMAGE_POSTGRES)
        env = []
        env.append({'name': 'CONTAINER_MODE', 'value': CONTAINER_MODE})
        env.append({'name': 'POSTGRES_HOST', 'value': POSTGRES_HOST})
        env.append({'name': 'POSTGRES_PORT', 'value': POSTGRES_PORT})
        env.append({'name': 'POSTGRES_PASSWORD', 'value': POSTGRES_PASSWORD})
        container.setdefault('env', env)
        template.setdefault('spec', {'containers': [container]})
        with open(f'{PYTHONPATH}/scripts/kubernetes/start_db_deployment.json', 'w') as f:
            f.write(json.dumps(content, indent=4))

    #redis service
    content = {}
    content.setdefault('apiVersion', 'v1')
    content.setdefault('kind', 'Service')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', REDIS_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', REDIS_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('type', 'NodePort')
    ports = spec.setdefault('ports', [])
    ports.append({'port': int(REDIS_PORT), 'targetPort': int(REDIS_PORT), 'nodePort': int(REDIS_EXT_PORT)})
    spec.setdefault('selector', {'name': REDIS_HOST})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_redis_service.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #redis deployment
    content = {}
    content.setdefault('apiVersion', 'apps/v1')      
    content.setdefault('kind', 'Deployment')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', REDIS_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', REDIS_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('replicas', 1)
    spec.setdefault('selector', {'matchLabels':{'name': REDIS_HOST}})
    template = spec.setdefault('template', {})
    template.setdefault('metadata', {'labels': {'name': REDIS_HOST}})
    container = {}
    container.setdefault('name', REDIS_HOST)
    container.setdefault('image', IMAGE_REDIS)
    template.setdefault('spec', {'containers': [container]})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_redis_deployment.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #admin deployment
    content = {}
    content.setdefault('apiVersion', 'apps/v1')      
    content.setdefault('kind', 'Deployment')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', ADMIN_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', ADMIN_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('replicas', 1)
    spec.setdefault('selector', {'matchLabels': {'name': ADMIN_HOST}})
    template = spec.setdefault('template', {})
    template.setdefault('metadata', {'labels': {'name': ADMIN_HOST}})
    container = {}
    container.setdefault('name', ADMIN_HOST)
    container.setdefault('image', f'{RAFIKI_IMAGE_ADMIN}:{RAFIKI_VERSION}')
    if CONTAINER_MODE == 'DEV':
        container.setdefault('volumeMounts', [{'name': ADMIN_HOST, 'mountPath': '/var/run/docker.sock'}, {'name': 'admin-log', 'mountPath': DOCKER_WORKDIR_PATH}])
        template.setdefault('spec', {'containers': [container], 'volumes': [{'name': ADMIN_HOST, 'hostPath': {'path': '/var/run/docker.sock'}}, \
                                    {'name': 'admin-log', 'hostPath': {'path': os.getenv('PWD', '')}}]})
    else:
        container.setdefault('volumeMounts', [{'name': 'work-path', 'mountPath': f'{DOCKER_WORKDIR_PATH}/{DATA_DIR_PATH}'}, \
                                              {'name': 'param-path', 'mountPath': f'{DOCKER_WORKDIR_PATH}/{PARAMS_DIR_PATH}'}, \
                                              {'name': 'log-path', 'mountPath': f'{DOCKER_WORKDIR_PATH}/{LOGS_DIR_PATH}'}, \
                                              {'name': ADMIN_HOST, 'mountPath': '/var/run/docker.sock'}])
        template.setdefault('spec', {'containers': [container], 'volumes': [{'name': 'work-path', 'hostPath': {'path': f'{HOST_WORKDIR_PATH}/{DATA_DIR_PATH}'}}, \
                                    {'name': 'param-path', 'hostPath': {'path': f'{HOST_WORKDIR_PATH}/{PARAMS_DIR_PATH}'}}, \
                                    {'name': 'log-path', 'hostPath': {'path': f'{HOST_WORKDIR_PATH}/{LOGS_DIR_PATH}'}}, \
                                    {'name': ADMIN_HOST, 'hostPath': {'path': '/var/run/docker.sock'}}]})
    env = []
    env.append({'name': 'POSTGRES_HOST', 'value': POSTGRES_HOST})
    env.append({'name': 'POSTGRES_PORT', 'value': POSTGRES_PORT})
    env.append({'name': 'POSTGRES_USER', 'value': POSTGRES_USER})
    env.append({'name': 'POSTGRES_DB', 'value': POSTGRES_DB})
    env.append({'name': 'POSTGRES_PASSWORD', 'value': POSTGRES_PASSWORD})
    env.append({'name': 'SUPERADMIN_PASSWORD', 'value': SUPERADMIN_PASSWORD})
    env.append({'name': 'ADMIN_HOST', 'value': ADMIN_HOST})
    env.append({'name': 'ADMIN_PORT', 'value': ADMIN_PORT})
    env.append({'name': 'REDIS_HOST', 'value': REDIS_HOST})
    env.append({'name': 'REDIS_PORT', 'value': REDIS_PORT})
    env.append({'name': 'KAFKA_HOST', 'value': KAFKA_HOST})
    env.append({'name': 'KAFKA_PORT', 'value': KAFKA_PORT})
    env.append({'name': 'PREDICTOR_PORT', 'value': PREDICTOR_PORT})
    env.append({'name': 'RAFIKI_ADDR', 'value': RAFIKI_ADDR})
    env.append({'name': 'RAFIKI_IMAGE_WORKER', 'value': RAFIKI_IMAGE_WORKER})
    env.append({'name': 'RAFIKI_IMAGE_PREDICTOR', 'value': RAFIKI_IMAGE_PREDICTOR})
    env.append({'name': 'RAFIKI_VERSION', 'value': RAFIKI_VERSION})
    env.append({'name': 'DOCKER_WORKDIR_PATH', 'value': DOCKER_WORKDIR_PATH})
    env.append({'name': 'WORKDIR_PATH', 'value': DOCKER_WORKDIR_PATH})
    env.append({'name': 'HOST_WORKDIR_PATH', 'value': HOST_WORKDIR_PATH})
    env.append({'name': 'DATA_DIR_PATH', 'value': DATA_DIR_PATH})
    env.append({'name': 'PARAMS_DIR_PATH', 'value': PARAMS_DIR_PATH})
    env.append({'name': 'LOGS_DIR_PATH', 'value': LOGS_DIR_PATH})
    env.append({'name': 'APP_MODE', 'value': APP_MODE})
    env.append({'name': 'CONTAINER_MODE', 'value': CONTAINER_MODE})
    container.setdefault('env', env)
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_admin_deployment.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #admin service
    content = {}
    content.setdefault('apiVersion', 'v1')
    content.setdefault('kind', 'Service')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', ADMIN_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', ADMIN_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('type', 'NodePort')
    ports = spec.setdefault('ports', [])
    ports.append({'port': int(ADMIN_PORT), 'targetPort': int(ADMIN_PORT), 'nodePort': int(ADMIN_EXT_PORT)})
    spec.setdefault('selector', {'name': ADMIN_HOST})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_admin_service.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #web service
    content = {}
    content.setdefault('apiVersion', 'v1')
    content.setdefault('kind', 'Service')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', WEB_ADMIN_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', WEB_ADMIN_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('type', 'NodePort')
    ports = spec.setdefault('ports', [])
    ports.append({'port': int(3001), 'targetPort': int(3001), 'nodePort': int(WEB_ADMIN_EXT_PORT)})
    spec.setdefault('selector', {'name': WEB_ADMIN_HOST})
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_web_admin_service.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

    #web deployment
    content = {}
    content.setdefault('apiVersion', 'apps/v1')      
    content.setdefault('kind', 'Deployment')
    metadata = content.setdefault('metadata', {})
    metadata.setdefault('name', WEB_ADMIN_HOST)
    labels = metadata.setdefault('labels', {})
    labels.setdefault('name', WEB_ADMIN_HOST)
    spec = content.setdefault('spec', {})
    spec.setdefault('replicas', 1)
    spec.setdefault('selector', {'matchLabels': {'name': WEB_ADMIN_HOST}})
    template = spec.setdefault('template', {})
    template.setdefault('metadata', {'labels': {'name': WEB_ADMIN_HOST}})
    container = {}
    container.setdefault('name', WEB_ADMIN_HOST)
    container.setdefault('image', f'{RAFIKI_IMAGE_WEB_ADMIN}:{RAFIKI_VERSION}')
    template.setdefault('spec', {'containers': [container]})
    env = []
    env.append({'name': 'RAFIKI_ADDR', 'value': RAFIKI_ADDR})
    env.append({'name': 'ADMIN_EXT_PORT', 'value': ADMIN_EXT_PORT})
    container.setdefault('env', env)
    with open(f'{PYTHONPATH}/scripts/kubernetes/start_web_admin_deployment.json', 'w') as f:
        f.write(json.dumps(content, indent=4))

