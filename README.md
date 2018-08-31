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

## Using Rafiki with the Admin Python module

In terminal 3, first create an admin with the Rafiki Admin Python module:

```shell
source .env.sh
python
```

```python
from admin import Admin
admin = Admin()
admin.create_user(
    email='admin@rafiki',
    password='rafiki',
    user_type='ADMIN'
)
```

Authenticating as an user:

```python
user = admin.authenticate_user('admin@rafiki', 'rafiki')
```

Creating & viewing models:

```python
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
```python
admin.create_train_job(
    user_id=user['id'],
    budget_type='TRIAL_COUNT',
    budget_amount=10,
    app_name='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
    test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test'
)
admin.get_train_jobs(app_name='fashion_mnist_app')
```

As the worker generates trials, checking on the completed trials of the train job:
```shell
admin.get_trials_by_train_job(train_job_id=<train_job_id>)
```

Making a prediction with the best trial of an app:
```shell
python src/scripts/predict_with_best_trial.py
```

Creating a deployment job for an app after model training:
```shell
python src/scripts/create_inference_job.py <email> <password>
```

This example uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Using Rafiki over HTTP

In terminal 3, first create an admin with the Rafiki Admin Python module:

```shell
source .env.sh
python
```

```python
from admin import Admin
admin = Admin()
admin.create_user(
    email='admin@rafiki',
    password='rafiki',
    user_type='ADMIN'
)
```

Then, start the Rafiki Admin HTTP server:

```shell
python src/scripts/start_admin.py
```

In terminal 4, use the Rafiki Client Python module on the Python CLI.

Initilizing the client & logging in:

```python
from client import Client
client = Client()
client.login(email='admin@rafiki', password='rafiki')
```

Creating an user:

```python
client.create_user(
    email='app_developer@rafiki',
    password='app_developer',
    user_type='APP_DEVELOPER'
)
```

Creating & viewing models:

```python
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

```python
client.create_train_job(
    budget_type='TRIAL_COUNT',
    budget_amount=10,
    app_name='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
    test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test'
)
client.get_train_jobs(app_name='fashion_mnist_app')
```

Viewing best trials of an app:

```python
client.get_best_trials_by_app(app_name='fashion_mnist_app')
```

## Rafiki Admin HTTP Server REST API

To make calls to the HTTP endpoints, you'll need first authenticate with email & password against the `POST /tokens` endpoint to obtain an authentication token `token`, and subsequently add the `Authorization` header for every other call:

`Authorization: Bearer {{token}}`

The list of available HTTP endpoints & their request formats are available as a *Postman* collection in the root of this project.

### Creating a Model

For the `POST /models` endpoint, you'll need to first serialize the model:

```python
from common import serialize_model_to_file
from model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel
model_inst = SingleHiddenLayerTensorflowModel()
serialize_model_to_file(model_inst, out_file_path='model.pickle')
```

Then, together with the `name` & `task` fields, upload the output serialized model file as the `model_serialized` field of a multi-part form data request.

## Credits

Original Auto-Tune Models (ATM) project: https://github.com/HDI-Project/ATM