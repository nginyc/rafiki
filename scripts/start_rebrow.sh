REBROW_PORT=5001

docker run --rm --name rafiki_rebrow \
  --network $DOCKER_NETWORK \
  -e REBROW_PORT=$REBROW_PORT \
  -p $REBROW_PORT:$REBROW_PORT \
  marian/rebrow