usage()  {
  echo "Usage: $0 <service_name> <rafiki_service_id> <rafiki_service_type>"
  exit 1
}

if [ $# -ne 3 ] ; then
    usage
fi

docker service create --name $1 \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e ADMIN_HOST=$ADMIN_HOST \
  -e ADMIN_PORT=$ADMIN_PORT \
  -e ADVISOR_HOST=$ADVISOR_HOST \
  -e ADVISOR_PORT=$ADVISOR_PORT \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -e RAFIKI_SERVICE_TYPE=$3 \
  -e RAFIKI_SERVICE_ID=$2 \
  -e LOGS_FOLDER_PATH=$LOGS_FOLDER_PATH \
  --mount type=bind,src=$LOGS_FOLDER_PATH,dst=$LOGS_FOLDER_PATH \
  $RAFIKI_IMAGE_WORKER:$RAFIKI_VERSION $2
