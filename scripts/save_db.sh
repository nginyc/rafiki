DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

# Check if dump file exists
if [ -f $DUMP_FILE ]
then 
    read -p "Database dump file exists at $DUMP_FILE. Override it? (y/n) " ok
    if [ $ok = "n" ] 
    then 
        echo "Not dumping database!" 
    else
        echo "Dumping database to $DUMP_FILE..." 
        docker exec $POSTGRES_HOST pg_dump -U postgres --if-exists --clean $POSTGRES_DB > $DUMP_FILE
    fi
fi
