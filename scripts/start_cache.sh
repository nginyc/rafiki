LOG_FILE_PATH=$PWD/logs/start_cache.log

source ./scripts/utils.sh

title "Starting Rafiki's Cache..."
(docker run --rm --name $REDIS_HOST \
  --network $DOCKER_NETWORK \
  -p $REDIS_EXT_PORT:$REDIS_PORT \
  $IMAGE_REDIS \
  &> $LOG_FILE_PATH) &

ensure_stable "Rafiki's Cache" $LOG_FILE_PATH
