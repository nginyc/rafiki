DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

echo "Dumping database to $DUMP_FILE..." 
docker exec $POSTGRES_HOST pg_dump -U postgres --if-exists --clean $POSTGRES_DB > $DUMP_FILE