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

# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

# Ensure docker container is stable
ensure_stable()
{
    log_file_path=$2
    sleep_time=$3
    echo "Waiting for ${sleep_time}s for $1 to stabilize..."
    sleep $sleep_time
    if ps -p $! > /dev/null
    then
        echo "$1 is running"
    else
        echo "Error running $1"
        echo "Maybe $1 hasn't previously been stopped - try running scripts/stop.sh?"
        if ! [ -z "$log_file_path" ]
        then
            echo "Check the logs at $log_file_path"
        fi
        exit 1
    fi
}

# Delete a folder or file with confirmation 
delete_path()
{
    path=$1
    read -p "Confirm remove $path? (y/n) " ok
    if [ $ok = "y" ] 
    then 
        echo "Removing $path..." 
        rm -rf $path
    fi
}