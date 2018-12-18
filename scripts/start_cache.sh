docker run --rm --name $REDIS_HOST \
  --network $DOCKER_NETWORK \
  -p $REDIS_EXT_PORT:$REDIS_PORT \
  $IMAGE_REDIS