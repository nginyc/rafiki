source ./scripts/utils.sh

# Read from shell configuration file
source ./.env.sh

# Create Docker swarm for Rafiki
bash ./scripts/create_docker_swarm.sh

# Pull images from Docker Hub
bash ./scripts/pull_images.sh || exit 1

# Start whole Rafiki stack
bash ./scripts/start_db.sh || exit 1
bash ./scripts/load_db.sh || exit 1
bash ./scripts/start_cache.sh || exit 1
bash ./scripts/start_admin.sh || exit 1
bash ./scripts/start_advisor.sh || exit 1
bash ./scripts/start_web_admin.sh || exit 1

title "Installing any dependencies..."
pip install -r ./rafiki/client/requirements.txt

echo "To use Rafiki, use Rafiki Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/docs/src/user/quickstart.html"
echo "To configure Rafiki, refer to Rafiki's developer docs at https://nginyc.github.io/rafiki/docs/latest/docs/src/dev/setup.html"
