DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

# Check if dump file exists
if [ -f $DUMP_FILE ]
then 
    echo "Loading database dump at $DUMP_FILE..." 
    cat $DUMP_FILE | docker exec -i $POSTGRES_HOST psql -U $POSTGRES_USER > /dev/null
else
    echo "No database dump file found." 
fi