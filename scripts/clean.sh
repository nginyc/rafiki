# Delete database dump
rm $POSTGRES_DUMP_FILE_PATH

# Clean all files within data, logs and params folder
rm -rf $DATA_WORKDIR_PATH/*
rm -rf $PARAMS_WORKDIR_PATH/*
rm -rf $LOGS_WORKDIR_PATH/*
