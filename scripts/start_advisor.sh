LOG_FILE_PATH=$PWD/logs/start_advisor.log

# Mount whole project folder with code for dev for shorter iterations
if [ $APP_MODE = "DEV" ]; then
  VOLUME_MOUNTS="-v $PWD:$DOCKER_WORKDIR_PATH"
else
  VOLUME_MOUNTS="-v $HOST_WORKDIR_PATH/$LOGS_DIR_PATH:$DOCKER_WORKDIR_PATH/$LOGS_DIR_PATH"
fi

source ./scripts/utils.sh

title "Starting Rafiki's Advisor..."
(docker run --rm --name $ADVISOR_HOST \
  --network $DOCKER_NETWORK \
  -e WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
  -e LOGS_DIR_PATH=$LOGS_DIR_PATH \
  $VOLUME_MOUNTS \
  -p $ADVISOR_EXT_PORT:$ADVISOR_PORT \
  $RAFIKI_IMAGE_ADVISOR:$RAFIKI_VERSION \
  &> $LOG_FILE_PATH) &

ensure_stable "Rafiki's Advisor" $LOG_FILE_PATH 10