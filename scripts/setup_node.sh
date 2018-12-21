echo "Listing nodes in Docker Swarm..."
docker node ls

read -p "Hostname of node to configure? " hostname
while true; do
    read -p "No. of GPUs? (0-9) " gpus
    case $gpus in
        [0-9] ) break;;
        * ) echo "Please answer a integer from 0-9.";;
    esac
done
docker node update --label-add gpu=$gpus $hostname