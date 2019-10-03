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

source ./scripts/utils.sh

pull_image()
{
    if [[ ! -z $(docker images -q $1) ]]
    then
        echo "$1 already exists locally"
    else 
        docker pull $1 || exit 1 
    fi
}

title "Pulling images..."
echo "Pulling images required by SingaAuto from Docker Hub..."
pull_image $IMAGE_POSTGRES
pull_image $IMAGE_REDIS
pull_image $IMAGE_KAFKA
pull_image $IMAGE_ZOOKEEPER
pull_image $SINGAAUTO_IMAGE_ADMIN:$SINGAAUTO_VERSION
pull_image $SINGAAUTO_IMAGE_WORKER:$SINGAAUTO_VERSION
pull_image $SINGAAUTO_IMAGE_PREDICTOR:$SINGAAUTO_VERSION
pull_image $SINGAAUTO_IMAGE_WEB_ADMIN:$SINGAAUTO_VERSION