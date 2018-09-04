MODULE_NAME=rafiki_daemon

docker build -t $MODULE_NAME -f ./deploy/daemon.Dockerfile $PWD
docker run --rm --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -e LOGS_FOLDER_PATH=$LOGS_FOLDER_PATH \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  $MODULE_NAME