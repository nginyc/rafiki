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

# Read from shell configuration file
source ./.env.sh

# Create Docker swarm for Rafiki
bash ./scripts/create_docker_swarm.sh

# Pull images from Docker Hub
bash ./scripts/pull_images.sh || exit 1

# Start whole Rafiki stack
# Skip starting & loading DB if DB is already running
if is_running $POSTGRES_HOST
then
  echo "Detected that Rafiki's DB is already running!"
else
    bash ./scripts/start_db.sh || exit 1
    bash ./scripts/load_db.sh || exit 1
fi
bash ./scripts/start_cache.sh || exit 1
bash ./scripts/start_admin.sh || exit 1
bash ./scripts/start_web_admin.sh || exit 1

echo "To use Rafiki, use Rafiki Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/docs/src/user/quickstart.html"
echo "To configure Rafiki, refer to Rafiki's developer docs at https://nginyc.github.io/rafiki/docs/latest/docs/src/dev/setup.html"
