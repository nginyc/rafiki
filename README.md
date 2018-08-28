# Rafiki2

## Installation

1. Install Docker

2. Install Python 3.6 & install the project's Python dependencies by running `pip install -r ./requirements.txt`.

## Setting Up the Stack

Create a .env.sh at root of project:
```
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_USER=rafiki
export POSTGRES_DB=rafiki
export POSTGRES_PASSWORD=rafiki
export PYTHONPATH=$PWD/src
```

Start the database in terminal 1:

```shell
source .env.sh
bash scripts/start_db.sh
```

Start a single worker in terminal 2:

```shell
source .env.sh
python src/scripts/start_worker.py
```

## Using Rafiki with Python

In terminal 3, first run:

```shell
source .env.sh
```

Creating an app:
```shell
python src/scripts/create_app.py
```

Creating a model:
```shell
python src/scripts/create_model.py
```

Creating a train job for the newly created app:
```shell
python src/scripts/create_model.py
```

As the worker generates trials, checking on the trials of the train job:
```shell
python src/scripts/get_train_job_trials.py <train_job_id>
```

This example uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Using Rafiki over HTTP

Start the admin HTTP server in terminal 3:

```shell
source .env.sh
python src/scripts/start_admin.py
```

The list of available HTTP endpoints are available as a *Postman* collection in the root of this project.

For the HTTP endpoint to create a model, you'll need to first serialize the model:

```shell
python src/scripts/serialize_model.py <model_class_name>
```

## Credits

Original Auto-Tune Models (ATM) project: https://github.com/HDI-Project/ATM