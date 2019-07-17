LOG_FILE_PATH=$PWD/logs/start_kafka.log

source ./scripts/utils.sh

ensure_kafka()
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

title "Starting Rafiki's Kafka..."

(docker run --rm --name $KAFKA_HOST \
  --network $DOCKER_NETWORK \
  -e KAFKA_ZOOKEEPER_CONNECT=$ZOOKEEPER_HOST:$ZOOKEEPER_PORT \
  -e KAFKA_ADVERTISED_HOST_NAME=$KAFKA_HOST \
  -e KAFKA_ADVERTISED_PORT=$KAFKA_PORT \
  -p $KAFKA_EXT_PORT:$KAFKA_PORT \
  -d $IMAGE_KAFKA \
  &> $LOG_FILE_PATH) &

ensure_kafka "Rafiki's Kafka" $LOG_FILE_PATH 30