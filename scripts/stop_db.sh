LOG_FILEPATH=$PWD/logs/stop.log

source ./scripts/utils.sh

title "Dumping database..." 
bash ./scripts/save_db.sh

# If database dump previously failed, prompt whether to continue script
if [ $? -ne 0 ]
then
    if ! prompt "Failed to dump database. Continue?"
    then
        exit 1
    fi
fi

title "Stopping Rafiki's DB..."
docker rm -f $POSTGRES_HOST || echo "Failed to stop Rafiki's DB"
