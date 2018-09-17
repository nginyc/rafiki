MODULE_NAME=rafiki_rebrow

docker run --rm --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e REBROW_PORT=$REBROW_PORT \
  -p $REBROW_PORT:$REBROW_PORT \
  marian/rebrow