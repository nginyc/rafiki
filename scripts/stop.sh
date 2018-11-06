source ./.env.sh

LOG_FILEPATH=$LOGS_FOLDER_PATH/stop.log

# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

title "Stopping Rafiki's DB..."
docker rm -f $POSTGRES_HOST || echo "Failed to stop Rafiki's DB"

title "Stopping Rafiki's Cache..."
docker rm -f $REDIS_HOST || echo "Failed to stop Rafiki's Cache"

title "Stopping Rafiki's Admin..."
docker rm -f $ADMIN_HOST || echo "Failed to stop Rafiki's Admin"

title "Stopping Rafiki's Advisor..."
docker rm -f $ADVISOR_HOST || echo "Failed to stop Rafiki's Advisor"

title "Stopping Rafiki's Admin Web..."
docker rm -f $ADMIN_WEB_HOST || echo "Failed to stop Rafiki's Admin Web"

echo "You'll need to destroy your machine's Docker swarm & Rafiki's logs folder at $LOGS_FOLDER_PATH manually"

