LOG_FILE_PATH=$PWD/logs/start_web_admin.log

source ./scripts/utils.sh

title "Starting Rafiki's Web Admin..."
(docker run --rm --name $WEB_ADMIN_HOST \
  --network $DOCKER_NETWORK \
  -e RAFIKI_ADDR=$RAFIKI_ADDR \
  -e ADMIN_EXT_PORT=$ADMIN_EXT_PORT \
  -p $WEB_ADMIN_EXT_PORT:3001 \
  $RAFIKI_IMAGE_WEB_ADMIN:$RAFIKI_VERSION \
  &> $LOG_FILE_PATH) &

ensure_stable "Rafiki's Web Admin" $LOG_FILE_PATH 10
