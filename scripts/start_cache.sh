DB_IMAGE=redis:5.0-rc

docker run --rm --name $REDIS_HOST \
  --network $DOCKER_NETWORK \
  -e REDIS_HOST=$REDIS_HOST \
  -e REDIS_PORT=$REDIS_PORT \
  -p $REDIS_PORT:$REDIS_PORT \
  $DB_IMAGE