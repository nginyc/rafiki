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

DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

source ./scripts/utils.sh

# Check if dump file exists
if [ -f $DUMP_FILE ]
then 
    if ! prompt "Database dump file exists at $DUMP_FILE. Override it?"
    then 
        echo "Not dumping database!" 
        exit 0
    fi
fi

echo "Dumping database to $DUMP_FILE..." 
docker exec $POSTGRES_HOST pg_dump -U postgres --if-exists --clean $POSTGRES_DB > $DUMP_FILE
