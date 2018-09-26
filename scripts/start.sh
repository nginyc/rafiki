# Read from shell configuration file
source ./.env.sh

LOG_FILEPATH=$LOGS_FOLDER_PATH/start.log
FILE_DIR=$(dirname "$0")

# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

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

# Create logs folder

title "Creating Rafiki's logs folder..."
bash $FILE_DIR/create_logs_folder.sh

# Pull images from Docker Hub

title "Pulling images for Rafiki from Docker Hub..."
bash $FILE_DIR/pull_images.sh

# Start whole Rafiki stack

title "Starting Rafiki's DB..."
(bash $FILE_DIR/start_db.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's DB"

title "Starting Rafiki's Cache..."
(bash $FILE_DIR/start_cache.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Cache"

title "Starting Rafiki's Admin..."
(bash $FILE_DIR/start_admin.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Admin"

title "Starting Rafiki's Advisor..."
(bash $FILE_DIR/start_advisor.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Advisor"

echo "To use Rafiki, use Rafiki Client in the Python CLI"
echo "Refer to Rafiki's docs at https://nginyc.github.io/rafiki2/docs/"
