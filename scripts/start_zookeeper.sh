LOG_FILE_PATH=$PWD/logs/start_zookeeper.log

source ./scripts/utils.sh

ensure_zookeeper()
{
    log_file_path=$2
    sleep_time=$3
    echo "Waiting for ${sleep_time}s for $1 to stabilize..."
    sleep $sleep_time
    if [ $? -eq 0 ]
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

title "Starting Rafiki's Zookeeper..."

(docker run --rm --name $ZOOKEEPER_HOST \
  --network $DOCKER_NETWORK \
  -p $ZOOKEEPER_EXT_PORT:$ZOOKEEPER_PORT \
  -d $IMAGE_ZOOKEEPER \
  &> $LOG_FILE_PATH) &
ensure_zookeeper "Rafiki's Zookeeper" $LOG_FILE_PATH 30