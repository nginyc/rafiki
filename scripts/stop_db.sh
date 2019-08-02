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

title "Dumping database..." 
bash ./scripts/save_db.sh

# If database dump previously failed, prompt whether to continue script
if [ $? -ne 0 ]
then
    if ! prompt "Failed to dump database. Continue?"
    then
        exit 1
    fi
fi

title "Stopping Rafiki's DB..."
docker rm -f $POSTGRES_HOST || echo "Failed to stop Rafiki's DB"
