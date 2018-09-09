MODULE_NAME=qfe

docker build -t $MODULE_NAME -f ./deploy/qfe.Dockerfile $PWD
docker run --rm --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e QFE_HOST=$QFE_HOST \
  -e QFE_PORT=$QFE_PORT \
  -e REDIS_HOST=$REDIS_HOST \
  -e REDIS_PORT=$REDIS_PORT \
  -e LOGS_FOLDER_PATH=$LOGS_FOLDER_PATH \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  -p $QFE_PORT:$QFE_PORT \
  $MODULE_NAME