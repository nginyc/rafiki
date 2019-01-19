# Read from shell configuration file
source ./.env.sh

pip install -r rafiki/test/requirements.txt
pytest -s --cov=rafiki $1