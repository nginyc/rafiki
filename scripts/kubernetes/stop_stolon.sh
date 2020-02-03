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

DB_CLUSTER_RUNNING_FILE=$HOST_WORKDIR_PATH/$RUN_DIR_PATH/DB-CLUSTER-RUNNING

title "Stopping Rafiki's Stolon Proxy Service..."
kubectl delete service stolon-proxy-service
sleep 5

title "Stopping Rafiki's Stolon Proxy..."
kubectl delete deployment stolon-proxy
sleep 5

title "Stopping Rafiki's Stolon Keeper..."
kubectl delete statefulset stolon-keeper
sleep 5

title "Stopping Rafiki's Stolon Sentinel..."
kubectl delete deployment stolon-sentinel
sleep 5

kubectl delete secret stolon

echo "Delete PVC..."
kubectl delete pvc database-stolon-keeper-0
kubectl delete pvc database-stolon-keeper-1
echo "Delete PV..."
kubectl delete pv database-pv-0
kubectl delete pv database-pv-1

echo "Remove databse running file $DB_CLUSTER_RUNNING_FILE"
rm -f $DB_CLUSTER_RUNNING_FILE
