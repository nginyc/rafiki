DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

source ./scripts/utils.sh

title "Maybe loading from database dump..." 

# Check if dump file exists
if [ -f $DUMP_FILE ]
then 
    echo "Loading database dump at $DUMP_FILE..." 
    cat $DUMP_FILE | docker exec -i $POSTGRES_HOST psql -U postgres --dbname $POSTGRES_DB > /dev/null
else
    echo "No database dump file found." 
fi