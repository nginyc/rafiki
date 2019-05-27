LOG_FILE_PATH=$PWD/logs/start_admin_web.log

source ./scripts/utils.sh

title "Starting Rafiki's Web Admin..."
(docker run --rm --name $ADMIN_WEB_HOST \
  --network $DOCKER_NETWORK \
  -e RAFIKI_ADDR=$RAFIKI_ADDR \
  -e ADMIN_EXT_PORT=$ADMIN_EXT_PORT \
  -p $ADMIN_WEB_EXT_PORT:3001 \
  $RAFIKI_IMAGE_ADMIN_WEB:$RAFIKI_VERSION \
  &> $LOG_FILE_PATH) &

ensure_stable "Rafiki's Web Admin" $LOG_FILE_PATH 10
