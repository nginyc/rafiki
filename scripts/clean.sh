# Read from shell configuration file
source ./.env.sh

# Clean all files within data, logs and params folder
rm -rf $WORKDIR_PATH/$DATA_DIR_PATH/*
rm -rf $WORKDIR_PATH/$LOGS_DIR_PATH/*
rm -rf $WORKDIR_PATH/$PARAMS_DIR_PATH/*