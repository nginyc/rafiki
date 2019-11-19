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

echo "Creating Database for Unit Test..."
kubectl create -f ./test/scripts/kubernetes/unit_test_database.yaml
sleep 20

echo "Creating Rafiki's PostgreSQL database & user..."
DB_PODNAME=$(kubectl get pod | grep rafiki-db-unittest)
DB_PODNAME=${DB_PODNAME:0:35}
kubectl exec $DB_PODNAME -- psql -U postgres -c "CREATE DATABASE rafiki"
kubectl exec $DB_PODNAME -- psql -U postgres -c "CREATE USER rafiki WITH PASSWORD 'rafiki'"
sleep 10

cp -f ./test/scripts/kubernetes/test.yaml.template      ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#TESTTYPE#rafiki-test-unit#g"  ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#RAFIKITESTIMAGE#$RAFIKI_IMAGE_TEST:$RAFIKI_VERSION#g"  ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#SUPERADMIN_PASSWORD_V#$SUPERADMIN_PASSWORD#g" ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#DOCKER_WORKDIR_PATH_V#$DOCKER_WORKDIR_PATH#g" ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#LOGS_DIR_PATH_V#$LOGS_DIR_PATH#g" ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#HOST_WORKDIR_PATH#$HOST_WORKDIR_PATH#g" ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#RAFIKI_ADDR_V#$ADMIN_HOST#g" ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#KAFKA_HOST_V#$KAFKA_HOST#g" ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#KAFKA_PORT_V#$KAFKA_PORT#g" ./test/scripts/kubernetes/unit_test.yaml
sed -ri "s#CMD#['bash', 'test/scripts/kubernetes/run_unit_test.sh']#g" ./test/scripts/kubernetes/unit_test.yaml
kubectl create -f ./test/scripts/kubernetes/unit_test.yaml
rm -rf ./test/scripts/kubernetes/unit_test.yaml

while (kubectl get job | grep rafiki-test-unit)
do
    echo "Waiting for Unit test finished!"
    sleep 30
done

if (cat $HOST_WORKDIR_PATH/$LOGS_DIR_PATH/test_unit.log | grep "failed");then
    echo "Unit test failed!"
    kubectl delete deployment rafiki-db-unittest
    kubectl delete svc rafiki-db-unittest
    exit 1
else
    echo "Unit test passed!"
    kubectl delete deployment rafiki-db-unittest
    kubectl delete svc rafiki-db-unittest
    exit 0
fi