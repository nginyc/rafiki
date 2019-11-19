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

source ./scripts/kubernetes/utils.sh

title "Starting Rafiki's DB..."

LOG_FILE_PATH=$PWD/logs/start_db_service.log
(kubectl create -f scripts/kubernetes/start_db_service.json \
&> $LOG_FILE_PATH) &
ensure_stable "Rafiki's DB Service" $LOG_FILE_PATH 20

LOG_FILE_PATH=$PWD/logs/start_db_deployment.log
(kubectl create -f scripts/kubernetes/start_db_deployment.json \
&> $LOG_FILE_PATH) &
ensure_stable "Rafiki's DB Deployment" $LOG_FILE_PATH 20

echo "Creating Rafiki's PostgreSQL database & user..."
DB_PODNAME=$(kubectl get pod | grep $POSTGRES_HOST)
DB_PODNAME=${DB_PODNAME:0:26}
kubectl exec $DB_PODNAME -c $POSTGRES_HOST -- psql -U postgres -c "CREATE DATABASE $POSTGRES_DB"
kubectl exec $DB_PODNAME -c $POSTGRES_HOST -- psql -U postgres -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD'"
