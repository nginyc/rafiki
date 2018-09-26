docker swarm init --advertise-addr $RAFIKI_IP_ADDRESS \
    || >&2 echo "Failed to init Docker swarm - continuing..."
docker network create $DOCKER_NETWORK -d overlay --attachable --scope=swarm \
    || >&2 echo  "Failed to create Docker network for swarm - continuing..."