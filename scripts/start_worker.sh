IMAGE_NAME=rafiki_worker

usage()  {
  echo "Usage: $0 <container_name>"
  exit 1
}

if [ $# -ne 1 ] ; then
    usage
fi

docker run --name $1 \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  $IMAGE_NAME
