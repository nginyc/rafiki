MODULE_NAME=rebrow

docker build -t $MODULE_NAME -f ./deploy/rebrow.Dockerfile $PWD
docker run --rm --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e REBROW_PORT=$REBROW_PORT \
  -e LOGS_FOLDER_PATH=$LOGS_FOLDER_PATH \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  -p $REBROW_PORT:$REBROW_PORT \
  $MODULE_NAME