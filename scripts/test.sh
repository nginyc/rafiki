LOG_FILE_PATH=$PWD/logs/test.log

# Read from shell configuration file
source ./.env.sh

pip install -r rafiki/test/requirements.txt
pytest -s --cov=rafiki $1 > $LOG_FILE_PATH || echo "Tests failed."
echo "Test results at $LOG_FILE_PATH"