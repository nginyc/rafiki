docker run --rm --name $REDIS_HOST \
  --network $DOCKER_NETWORK \
  -e REDIS_HOST=$REDIS_HOST \
  -e REDIS_PORT=$REDIS_PORT \
  -p $REDIS_PORT:$REDIS_PORT \
  $IMAGE_REDIS