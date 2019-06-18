LOG_FILE_PATH=$PWD/logs/test.log

pip install -r test/requirements.txt

echo 'Running tests...'
echo "Check test results at $LOG_FILE_PATH"
pytest -s -x $1 > $LOG_FILE_PATH || echo "Tests failed"
