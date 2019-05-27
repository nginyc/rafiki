DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

echo "Dumping database to $DUMP_FILE..." 
docker exec $POSTGRES_HOST pg_dumpall -U $POSTGRES_USER --clean > $DUMP_FILE