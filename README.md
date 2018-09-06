# Rafiki2

This example uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Installation

Prerequisites: Unix-like environment

1. Install Docker

2. Install Python 3.6

## Setting Up the Stack

Create a Docker Swarm e.g.:

```sh
docker swarm init
```

Create a custom overlay Docker network for Rafiki, scoped to the Docker Swarm:

```sh
docker network create rafiki -d overlay --attachable --scope=swarm
```

Create the file `.env.sh` at root of project:

```sh
export POSTGRES_HOST=rafiki_db
export POSTGRES_PORT=5432
export POSTGRES_USER=rafiki
export POSTGRES_DB=rafiki
export POSTGRES_PASSWORD=rafiki
export PYTHONPATH=$PWD/src
export APP_SECRET=rafiki
export DOCKER_NETWORK=rafiki
export LOGS_FOLDER_PATH=/var/log/rafiki
```

Setup the Rafiki logs directory by creating the directory `/var/log/rafiki/` and ensuring Docker has the permissions to mount it onto containers:

```sh
sudo mkdir /var/log/rafiki
sudo chmod 777 /var/log/rafiki
```

Start the database in terminal 1:

```sh
source .env.sh
bash scripts/start_db.sh
```

Build the base Rafiki worker image in Docker:

```sh
source .env.sh
bash scripts/build_worker_image.sh
```

## Using Rafiki over HTTP

In terminal 2, start the Rafiki Admin HTTP server:

```sh
bash scripts/start_admin.sh
```

In terminal 3, bash in the Docker container of Rafiki Admin to create a new admin with the Rafiki Admin Python module:

```sh
docker exec -it rafiki_admin bash
python
```

```py
from admin import Admin
admin = Admin()
admin.create_user(
    email='admin@rafiki',
    password='rafiki',
    user_type='ADMIN'
)
exit()
```

```sh
exit
```

On terminal 3, use the Rafiki Client Python module on the Python CLI. You'll need to install the Rafiki Client's Python dependencies by running `pip install -r ./src/client/requirements.txt`.

```sh
source .env.sh
python
```

Initilizing the client & logging in:

```py
from client import Client
client = Client()
client.login(email='admin@rafiki', password='rafiki')
```

Creating an user:

```py
client.create_user(
    email='app_developer@rafiki',
    password='app_developer',
    user_type='APP_DEVELOPER'
)
```

Creating & viewing models:

```py
from model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel
model_inst = SingleHiddenLayerTensorflowModel()
client.create_model(
    name='single_hidden_layer_tf',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    model_inst=model_inst
)
client.get_models()
```

Creating & viewing train jobs:

```py
client.create_train_job(
    app_name='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
    test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test'
)
client.get_train_jobs(app_name='fashion_mnist_app')
```

Viewing best trials of an app:

```py
client.get_best_trials_by_app(app_name='fashion_mnist_app')
```

## Rafiki Admin HTTP Server REST API

To make calls to the HTTP endpoints, you'll need first authenticate with email & password against the `POST /tokens` endpoint to obtain an authentication token `token`, and subsequently add the `Authorization` header for every other call:

`Authorization: Bearer {{token}}`

The list of available HTTP endpoints & their request formats are available as a *Postman* collection in the root of this project.

### Creating a Model

For the `POST /models` endpoint, you'll need to first serialize the model:

```py
from common import serialize_model_to_file
from model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel
model_inst = SingleHiddenLayerTensorflowModel()
serialize_model_to_file(model_inst, out_file_path='model.pickle')
```

Then, together with the `name` & `task` fields, upload the output serialized model file as the `model_serialized` field of a multi-part form data request.

## Using Rafiki with the Admin Python module

Use the Rafiki Admin Python module on the Python CLI. You'll need to install the Rafiki Admin's Python dependencies by running `pip install -r ./src/admin/requirements.txt`. You'll also need to make sure `POSTGRES_HOST` and `POSTGRES_PORT` are configured right to communicate directly to the DB.

```shell
python
```

Creating an user:

```py
from admin import Admin
admin = Admin()
admin.create_user(
    email='admin@rafiki',
    password='rafiki',
    user_type='ADMIN'
)
```

Authenticating as an user:

```py
user = admin.authenticate_user('admin@rafiki', 'rafiki')
```

Creating & viewing models:

```py
from common import serialize_model
from model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel
model = SingleHiddenLayerTensorflowModel()
model_serialized = serialize_model(model)
admin.create_model(
    user_id=user['id'],
    name='single_hidden_layer_tf',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    model_serialized=model_serialized
)
admin.get_models()
```

Creating a train job:

```py
admin.create_train_job(
    user_id=user['id'],
    app_name='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
    test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test'
)
admin.get_train_jobs(app_name='fashion_mnist_app')
```

As the worker generates trials, checking on the completed trials of the train job:

```sh
admin.get_trials_by_train_job(train_job_id=<train_job_id>)
```

## Troubleshooting

You can read all logs in the logs directory:

```sh
open /var/log/rafiki
```

By default, you can connect to the PostgreSQL using a PostgreSQL client (e.g [Postico](https://eggerapps.at/postico/)) with these credentials:

```sh
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=rafiki
POSTGRES_DB=rafiki
POSTGRES_PASSWORD=rafiki
```

## Credits

Original Auto-Tune Models (ATM) project: https://github.com/HDI-Project/ATM