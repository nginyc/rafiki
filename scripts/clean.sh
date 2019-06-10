# Clean all files within data & logs folders
rm -rf $DATA_WORKDIR_PATH/*
rm -rf $LOGS_WORKDIR_PATH/*

# Delete SQL dump file
rm $POSTGRES_DUMP_FILE_PATH