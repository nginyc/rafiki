source ./scripts/utils.sh

pull_image()
{
    if [[ ! -z $(docker images -q $1) ]]
    then
        echo "$1 already exists locally"
    else 
        docker pull $1 || exit 1 
    fi
}

# Read from shell configuration file
source ./.env.sh

# Create Docker swarm for Rafiki
bash ./scripts/create_docker_swarm.sh

# Build local images
bash ./scripts/build_images.sh || exit 1

# Pull images from Docker Hub
echo "Pulling images required by Rafiki from Docker Hub..."
pull_image $IMAGE_POSTGRES
pull_image $IMAGE_REDIS

# Start whole Rafiki stack
bash ./scripts/start_db.sh || exit 1
bash ./scripts/load_db.sh || exit 1
bash ./scripts/start_cache.sh || exit 1
bash ./scripts/start_admin.sh || exit 1
bash ./scripts/start_web_admin.sh || exit 1

echo "To use Rafiki, use Rafiki Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/docs/src/user/quickstart.html"
echo "To configure Rafiki, refer to Rafiki's developer docs at https://nginyc.github.io/rafiki/docs/latest/docs/src/dev/setup.html"
