source ./scripts/utils.sh

title "Creating Docker swarm for Rafiki..."
docker swarm leave $1
docker swarm init --advertise-addr $DOCKER_SWARM_ADVERTISE_ADDR \
    || >&2 echo "Failed to init Docker swarm - continuing..."
docker network create $DOCKER_NETWORK -d overlay --attachable --scope=swarm \
    || >&2 echo  "Failed to create Docker network for swarm - continuing..."