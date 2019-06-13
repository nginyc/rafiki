source ./scripts/utils.sh

# Delete database dump
delete_path "$POSTGRES_DUMP_FILE_PATH"

# Clean all files within data, logs and params folder
delete_path "$DATA_WORKDIR_PATH/*"
delete_path "$PARAMS_WORKDIR_PATH/*"
delete_path "$LOGS_WORKDIR_PATH/*"