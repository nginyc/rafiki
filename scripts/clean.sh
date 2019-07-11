source ./scripts/utils.sh

# Read from shell configuration file
source ./.env.sh

# Delete database dump
delete_path "$POSTGRES_DUMP_FILE_PATH"

# Clean all files within data, logs and params folder
delete_path "$PWD/$DATA_DIR_PATH/*"
delete_path "$PWD/$PARAMS_DIR_PATH/*"
delete_path "$PWD/$LOGS_DIR_PATH/*"
