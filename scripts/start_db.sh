LOG_FILE_PATH=$PWD/logs/start_db.log

source ./scripts/utils.sh

title "Starting Rafiki's DB..."
(docker run --rm --name $POSTGRES_HOST \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -p $POSTGRES_EXT_PORT:$POSTGRES_PORT \
  $IMAGE_POSTGRES \
  &> $LOG_FILE_PATH) &

ensure_stable "Rafiki's DB" $LOG_FILE_PATH 20

echo "Creating Rafiki's PostgreSQL database & user..."
docker exec $POSTGRES_HOST psql -U postgres -c "CREATE DATABASE $POSTGRES_DB"
docker exec $POSTGRES_HOST psql -U postgres -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD'"