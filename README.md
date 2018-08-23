# Rafiki2

## Installation

1. Install Docker

2. Install Python 3.6 & install the project's Python dependencies by running `pip install -r ./requirements.txt`.

## Running the Stack

Create a .env.sh at root of project:
```
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_USER=rafiki
export POSTGRES_DB=rafiki
export POSTGRES_PASSWORD=rafiki
export PYTHONPATH=$PWD/src
```

Run in terminal 1:

```shell
bash scripts/start_db.sh
```

Run in terminal 2:

```
source .env.sh
python src/scripts/create_app.py
python src/scripts/create_model.py
python src/scripts/create_train_job.py
```

Run in terminal 3:

```shell
python src/scripts/start_worker.py
```

As training occurs, view the app's status with:

```
python src/scripts/get_app_status.py
```

This example uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Credits

Original Auto-Tune Models (ATM) project: https://github.com/HDI-Project/ATM