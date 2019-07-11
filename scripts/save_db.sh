DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

source ./scripts/utils.sh

# Check if dump file exists
if [ -f $DUMP_FILE ]
then 
    if ! prompt "Database dump file exists at $DUMP_FILE. Override it?"
    then 
        echo "Not dumping database!" 
        exit 0
    fi
fi

echo "Dumping database to $DUMP_FILE..." 
docker exec $POSTGRES_HOST pg_dump -U postgres --if-exists --clean $POSTGRES_DB > $DUMP_FILE
