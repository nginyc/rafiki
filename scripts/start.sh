LOG_FILEPATH=$PWD/logs/start.log
FILE_DIR=$(dirname "$0")

source ./scripts/utils.sh

# Read from shell configuration file
source ./.env.sh

ensure_stable()
{
    echo "Waiting for 10s for $1 to stablize..."
    sleep 10
    if ps -p $! > /dev/null
    then
        echo "$1 is running"
    else
        echo "Error running $1"
        echo "Check the logs at $LOG_FILEPATH"
        exit 1
    fi
}

# Create Docker swarm for Rafiki

title "Creating Docker swarm for Rafiki..."
bash $FILE_DIR/create_docker_swarm.sh

# Pull images from Docker Hub

title "Pulling images for Rafiki from Docker Hub..."
bash $FILE_DIR/pull_images.sh

# Start whole Rafiki stack

title "Starting Rafiki's DB..."
(bash $FILE_DIR/start_db.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's DB"

title "Maybe loading from database dump..." 
bash $FILE_DIR/load_db.sh

title "Starting Rafiki's Cache..."
(bash $FILE_DIR/start_cache.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Cache"

title "Starting Rafiki's Admin..."
(bash $FILE_DIR/start_admin.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Admin"

title "Starting Rafiki's Advisor..."
(bash $FILE_DIR/start_advisor.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Advisor"

title "Starting Rafiki's Admin Web..."
(bash $FILE_DIR/start_admin_web.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Admin Web"

title "Installing any dependencies..."
pip install -r ./rafiki/client/requirements.txt

echo "To use Rafiki, use Rafiki Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/docs/src/user/quickstart.html"
echo "To configure Rafiki, refer to Rafiki's developer docs at https://nginyc.github.io/rafiki/docs/latest/docs/src/dev/setup.html"
