IMAGE_NAME=rafiki_worker

usage()  {
  echo "Usage: $0 <service_name> <worker_id>"
  exit 1
}

if [ $# -ne 2 ] ; then
    usage
fi

docker service create --name $1 \
  --rm --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  $IMAGE_NAME $2
