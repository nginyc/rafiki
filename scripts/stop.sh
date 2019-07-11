LOG_FILEPATH=$PWD/logs/stop.log

source ./scripts/utils.sh

# Read from shell configuration file
source ./.env.sh

title "Stopping any existing jobs..."
python ./scripts/stop_all_jobs.py

# Prompt if should stop DB
if prompt "Should stop Rafiki's DB?"
then
    bash scripts/stop_db.sh || exit 1
else
    echo "Not stopping Rafiki's DB!"
fi

title "Stopping Rafiki's Cache..."
docker rm -f $REDIS_HOST || echo "Failed to stop Rafiki's Cache"

title "Stopping Rafiki's Admin..."
docker rm -f $ADMIN_HOST || echo "Failed to stop Rafiki's Admin"

title "Stopping Rafiki's Advisor..."
docker rm -f $ADVISOR_HOST || echo "Failed to stop Rafiki's Advisor"

title "Stopping Rafiki's Web Admin..."
docker rm -f $WEB_ADMIN_HOST || echo "Failed to stop Rafiki's Web Admin"

echo "You'll need to destroy your machine's Docker swarm manually"

