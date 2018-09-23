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
    echo "Waiting for 5s for $1 to stablize..."
    sleep 5
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

# Build Rafiki's images

title "Building Rafiki DB's image..."
bash $FILE_DIR/build_image_db.sh || exit 1 
title "Building Rafiki Cache's image..."
bash $FILE_DIR/build_image_cache.sh || exit 1 
title "Building Rafiki Admin's image..."
bash $FILE_DIR/build_image_admin.sh || exit 1 
title "Building Rafiki Advisor's image..."
bash $FILE_DIR/build_image_advisor.sh || exit 1 
title "Building Rafiki Model's image..."
bash $FILE_DIR/build_image_model.sh || exit 1 
title "Building Rafiki Query Frontend's image..."
bash $FILE_DIR/build_image_query_frontend.sh || exit 1 
echo "Finished building all Rafiki's images successfully!"

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
