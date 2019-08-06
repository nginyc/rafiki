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

source ./scripts/kubernetes/utils.sh

title "Stopping any existing jobs..."
python ./scripts/stop_all_jobs.py

title "Stopping Rafiki's Web Admin Deployment..."
kubectl delete deployment $WEB_ADMIN_HOST || echo "Failed to stop Rafiki's Web Admin Deployment"

title "Stopping Rafiki's Admin Deployment..."
kubectl delete deployment $ADMIN_HOST || echo "Failed to stop Rafiki's Admin Deployment"

title "Stopping Rafiki's Redis Deployment..."
kubectl delete deployment $REDIS_HOST || echo "Failed to stop Rafiki's Redis Deployment"

title "Stopping Rafiki's Kafka Deployment..."
kubectl delete deployment $KAFKA_HOST || echo "Failed to stop Rafiki's Kafka Deployment"

title "Stopping Rafiki's Zookeeper Deployment..."
kubectl delete deployment $ZOOKEEPER_HOST || echo "Failed to stop Rafiki's Zookeeper Deployment"

title "Stopping Rafiki's Web Admin Service..."
kubectl delete service $WEB_ADMIN_HOST || echo "Failed to stop Rafiki's Web Admin Service"

title "Stopping Rafiki's Admin Service..."
kubectl delete service $ADMIN_HOST || echo "Failed to stop Rafiki's Admin Service"

title "Stopping Rafiki's Redis Service..."
kubectl delete service $REDIS_HOST || echo "Failed to stop Rafiki's Redis Service"

title "Stopping Rafiki's Kafka Service..."
kubectl delete service $KAFKA_HOST || echo "Failed to stop Rafiki's Kafka Service"

title "Stopping Rafiki's Zookeeper Service..."
kubectl delete service $ZOOKEEPER_HOST || echo "Failed to stop Rafiki's Zookeeper Service"

# Prompt if should stop DB
if prompt "Should stop Rafiki's DB?"
then
    if [ "$CLUSTER_MODE" = "SINGLE" ]; then
        bash scripts/kubernetes/stop_db.sh || exit 1
    else
        bash scripts/kubernetes/stop_stolon.sh || exit 1
    fi
else
    echo "Not stopping Rafiki's DB!"
fi
