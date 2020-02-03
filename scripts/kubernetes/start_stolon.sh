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

DB_CLUSTER_RUNNING_FILE=$HOST_WORKDIR_PATH/$RUN_DIR_PATH/DB-CLUSTER-RUNNING
mkdir -p $HOST_WORKDIR_PATH/$RUN_DIR_PATH/

title "Starting Rafiki's DB..."
if [ -f $DB_CLUSTER_RUNNING_FILE ]; then
    echo "Rafiki's DB is already running, skip..."
    exit 0
fi

echo "Create PV..."
# With stolon, we use some default parameters to make nfs as pv, if your have another choice or want to change the default parameters,
# your should modify this script
bash scripts/kubernetes/create_nfs_pv.sh database-pv-0 $NFS_HOST_IP /home/rafiki/database/db0 100Gi ReadWriteOnce Retain pv database-pv-0
bash scripts/kubernetes/create_nfs_pv.sh database-pv-1 $NFS_HOST_IP /home/rafiki/database/db1 100Gi ReadWriteOnce Retain pv database-pv-1
echo "Create PVC..."
# PVC Name is Fixed
bash scripts/kubernetes/create_nfs_pvc.sh database-stolon-keeper-0 100Gi ReadWriteOnce pv database-pv-0
bash scripts/kubernetes/create_nfs_pvc.sh database-stolon-keeper-1 100Gi ReadWriteOnce pv database-pv-1

bash scripts/kubernetes/stolon/generate_stolon_yaml.sh
LOG_FILE_PATH=$PWD/logs/start_stolon_sentinel.log
(kubectl create -f scripts/kubernetes/stolon/stolon-sentinel.yaml \
&> $LOG_FILE_PATH) &
ensure_stable "Rafiki's Stolon Sentinel" $LOG_FILE_PATH 20

kubectl create -f scripts/kubernetes/stolon/secret.yaml

LOG_FILE_PATH=$PWD/logs/start_stolon_keeper.log
(kubectl create -f scripts/kubernetes/stolon/stolon-keeper.yaml \
&> $LOG_FILE_PATH) &
ensure_stable "Rafiki's Stolon Keeper" $LOG_FILE_PATH 20

LOG_FILE_PATH=$PWD/logs/start_stolon_proxy.log
(kubectl create -f scripts/kubernetes/stolon/stolon-proxy.yaml \
&> $LOG_FILE_PATH) &
ensure_stable "Rafiki's Stolon Proxy" $LOG_FILE_PATH 20

LOG_FILE_PATH=$PWD/logs/start_stolon_proxy_service.log
(kubectl create -f scripts/kubernetes/stolon/stolon-proxy-service.yaml \
&> $LOG_FILE_PATH) &
ensure_stable "Rafiki's Stolon Proxy Service" $LOG_FILE_PATH 10

DB_CLUSTER_INIT_FILE=$HOST_WORKDIR_PATH/$RUN_DIR_PATH/DB-CLUSTER-INIT-FLAG-CAN-NOT-REMOVE
if [ -f $DB_CLUSTER_INIT_FILE ]; then
    echo "The Database Cluster already initialized, don't need reinitialize ..."
    echo "Waiting for 60s for Rafiki's Stolon Cluster to stabilize..."
    sleep 60
else
    echo "Create databse initialized file $DB_CLUSTER_INIT_FILE"
    touch $DB_CLUSTER_INIT_FILE
    echo "Attention: The file `basename $DB_CLUSTER_INIT_FILE` can not be removed, otherwise your will 
        lost your databse data after exec $0 next time." > $DB_CLUSTER_INIT_FILE
    echo -e "\033[31m`cat $DB_CLUSTER_INIT_FILE` \033[0m"
    date >> $DB_CLUSTER_INIT_FILE

    kubectl run -i -t stolon-init-cluster --image=$RAFIKI_IMAGE_STOLON \
      --restart=Never --rm -- /usr/local/bin/stolonctl \
      --cluster-name=kube-stolon --store-backend=kubernetes \
      --kube-resource-kind=configmap init -y
    echo "Waiting for 60s for Rafiki's Stolon Cluster to stabilize..."
    sleep 60
    
    echo "Creating Rafiki's PostgreSQL database & user..."
    DB_SVC_IP=`kubectl get svc | grep stolon-proxy-service | awk '{print $3}'`
    STOLON_PASSWD=`echo $POSTGRES_STOLON_PASSWD|base64 -d`
    kubectl run -i -t stolon-create-db --image=$RAFIKI_IMAGE_STOLON \
      --restart=Never --env="PGPASSWORD=$STOLON_PASSWD" --rm -- psql -h $DB_SVC_IP postgres -U stolon -c "CREATE DATABASE $POSTGRES_DB"
    kubectl run -i -t stolon-create-user --image=$RAFIKI_IMAGE_STOLON \
      --restart=Never --env="PGPASSWORD=$STOLON_PASSWD" --rm -- psql -h $DB_SVC_IP postgres -U stolon -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD'"
fi

touch $DB_CLUSTER_RUNNING_FILE
echo "Create databse running file $DB_CLUSTER_RUNNING_FILE"
