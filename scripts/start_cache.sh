docker run --rm --name rafiki_cache \
  --network $DOCKER_NETWORK \
  -e REDIS_HOST=$REDIS_HOST \
  -e REDIS_PORT=$REDIS_PORT \
  -p $REDIS_PORT:$REDIS_PORT \
  $RAFIKI_IMAGE_CACHE