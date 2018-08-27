source .env

# Ensures python stdout appears immediately
export PYTHONUNBUFFERED=1

# Allow importing of python modules from project root
export PYTHONPATH=$PWD

python ./scripts/start_admin.py