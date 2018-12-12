# Determine whether node is Docker Swarm manager or worker
swarm_role=$1
if [ -z "$swarm_role" ] 
then
    while true; do
        read -p "Is this node a Docker Swarm manager running Rafiki? (y/n) " yn
        case $yn in
            [Yy]* ) swarm_role="manager" ; break;;
            [Nn]* ) swarm_role="worker" ; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi

# For workers, join Docker Swarm
if [ "$swarm_role" = "worker" ]
then
    read -p "IP address of Docker Swarm manager? " ip_addr    
    read -p "Docker Swarm join token? " join_token
    docker swarm leave $1
    docker swarm join --token $join_token $ip_addr
fi

# Add node label that specifies no. of GPUs
hostname=$(docker node inspect self | sed -n 's/"Hostname".*"\(.*\)".*/\1/p' | xargs)
while true; do
    read -p "No. of GPUs (0-9)? " gpus
    case $gpus in
        [0-9] ) break;;
        * ) echo "Please answer a integer from 0-9.";;
    esac
done
docker node update --label-add gpu=$gpus $hostname