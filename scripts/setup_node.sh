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
# Read from shell configuration file
source ./.env.sh

echo "Listing nodes in Docker Swarm..."
docker node ls

read -p "Hostname of node to configure? " hostname
while true; do
    read -p "GPU numbers available? (e.g. '' or '0,2') " gpus
    if echo "$gpus" | grep -Eq "^(([0-9],)*[0-9])?$"; then
        break
    fi
    echo "Please key in a comman-separated list of GPU numbers e.g. '' or '0,2'."
done
docker node update --label-add $DOCKER_NODE_LABEL_AVAILABLE_GPUS=$gpus $hostname
docker node update --label-add $DOCKER_NODE_LABEL_NUM_SERVICES=0 $hostname
