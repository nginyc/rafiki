echo "Listing nodes in Docker Swarm..."
docker node ls

read -p "Hostname of node to configure? " hostname
while true; do
    read -p "GPUs available? (e.g. '' or '0,2') " gpus
    if echo "$gpus" | grep -Eq "^(([0-9],)*[0-9])?$"; then
        break
    fi
    echo "Please key in a comman-separated list of GPU numbers e.g. '' or '0,2'."
done
docker node update --label-add available_gpus=$gpus $hostname
docker node update --label-add num_services=0 $hostname