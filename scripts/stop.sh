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

LOG_FILEPATH=$PWD/logs/stop.log
source ./scripts/utils.sh

# Read from shell configuration file
source ./.env.sh

title "Stopping any existing jobs..."
python ./scripts/stop_all_jobs.py

title "Dumping database..." 
bash ./scripts/save_db.sh

# If database dump previously failed, prompt whether to continue script
if [ $? -ne 0 ]
then
    read -p "Failed to dump database. Continue? (y/n) " ok
    if [ $ok = "n" ]
    then
        exit 1
    fi
fi

title "Stopping Rafiki's DB..."
docker rm -f $POSTGRES_HOST || echo "Failed to stop Rafiki's DB"

title "Stopping Rafiki's Cache..."
docker rm -f $REDIS_HOST || echo "Failed to stop Rafiki's Cache"

title "Stopping Rafiki's Admin..."
docker rm -f $ADMIN_HOST || echo "Failed to stop Rafiki's Admin"

title "Stopping Rafiki's Advisor..."
docker rm -f $ADVISOR_HOST || echo "Failed to stop Rafiki's Advisor"

title "Stopping Rafiki's Web Admin..."
docker rm -f $WEB_ADMIN_HOST || echo "Failed to stop Rafiki's Web Admin"

echo "You'll need to destroy your machine's Docker swarm manually"

