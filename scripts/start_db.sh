MODULE_NAME=rafiki_db

docker build -t $MODULE_NAME -f ./dockerfiles/db.Dockerfile $PWD
docker run --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -p 5433:$POSTGRES_PORT \
  $MODULE_NAME
 
# DB available on localhost at port 5433