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

# Delete database dump
delete_path "$POSTGRES_DUMP_FILE_PATH"

# Clean all files within data, logs and params folder
delete_path "$PWD/$DATA_DIR_PATH/*"
delete_path "$PWD/$PARAMS_DIR_PATH/*"
delete_path "$PWD/$LOGS_DIR_PATH/*"
