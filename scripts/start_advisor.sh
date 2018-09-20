MODULE_NAME=rafiki_advisor

docker build -t $MODULE_NAME -f ./dockerfiles/advisor.Dockerfile $PWD
docker run --rm --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e APP_SECRET=$APP_SECRET \
  -e LOGS_FOLDER_PATH=$LOGS_FOLDER_PATH \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  -p 8001:$ADMIN_PORT \
  $MODULE_NAME