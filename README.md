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

In terminal 3, first create an admin with the Python SDK:

```shell
source .env.sh
python src/scripts/create_user.py <email> <password> ADMIN
```

Creating an app:
```shell
python src/scripts/create_app.py <email> <password>
```

Creating a model:
```shell
python src/scripts/create_model.py <email> <password>
```

Creating a train job for the newly created app:
```shell
python src/scripts/create_train_job.py <email> <password>
```

As the worker generates trials, checking on the trials of the train job:
```shell
python src/scripts/get_train_job_trials.py <train_job_id>
```

Making a prediction with the best trial of the app:
```shell
python src/scripts/predict_with_best_trial.py
```

Creating a deployment job for the app after model training:
```shell
python src/scripts/create_deployment_job.py <email> <password>
```

This example uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Using Rafiki over HTTP

In terminal 3, create an admin with the Python SDK:

```shell
source .env.sh
python src/scripts/create_user.py <email> <password> ADMIN
```

Then, start the admin HTTP server:

```shell
python src/scripts/start_admin.py
```

To make calls to the HTTP endpoints, you'll need first authenticate with email & password against the `POST /tokens` endpoint to obtain an authentication token `token`, and subsequently add the `Authorization` header for every other call:

`Authorization: Bearer {{token}}`

The list of available HTTP endpoints & their request formats are available as a *Postman* collection in the root of this project.

### Creating a Model

For the `POST /models` endpoint, you'll need to first serialize the model:

```shell
python src/scripts/serialize_model.py <model_class_name>
```

Then, together with the `name` & `task` fields, upload this file as the `model_serialized` field of a multi-part form data request.

## Credits

Original Auto-Tune Models (ATM) project: https://github.com/HDI-Project/ATM