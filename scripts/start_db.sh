MODULE_NAME=rafiki_db

docker build -t $MODULE_NAME -f ./deploy/db.Dockerfile $PWD
docker run --name $MODULE_NAME \
  --network $DOCKER_NETWORK \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -p 5433:5432 \
  $MODULE_NAME
  