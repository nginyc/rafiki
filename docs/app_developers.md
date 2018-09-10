# Guide for App Developers

## Installation

1. Install Python 3.6

2. Setup & configure Raifki Client by running:

```sh
pip install -r ./src/client/requirements.txt
export PYTHONPATH=$PWD/src
```

## Using Rafiki

Use the Rafiki Client Python module on the Python CLI:

Logging in:

```py
from client import Client
client = Client()
client.login(email='app_developer@rafiki', password='rafiki')
```

```sh
{'user_id': 'eb273359-c74b-492b-80af-b9ea47ca959a', 'user_type': 'APP_DEVELOPER'}
```

Getting models by task:

```py
client.get_models_of_task(task='IMAGE_CLASSIFICATION_WITH_ARRAYS')
```

```sh
[{'datetime_created': 'Thu, 06 Sep 2018 04:38:48 GMT',
  'docker_image': 'rafiki_model',
  'name': 'single_hidden_layer_tf',
  'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
  'user_id': 'a8959685-6667-41d5-8f91-b195fda27f91'}]
```

Creating train jobs:

```py
client.create_train_job(
    app='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
    test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test',
    budget_type='MODEL_TRIAL_COUNT',
    budget_amount=3
)
```

```sh
{'id': 'd6a00b3a-317d-4804-ae1f-9a7d0e57d75f', 'app_version': 1}
```

Viewing train jobs of an app:

```py
client.get_train_jobs_of_app(app='fashion_mnist_app')
```

```sh
[{'app': 'fashion_mnist_app',
  'app_version': 1,
  'budget_amount': 10,
  'budget_type': 'MODEL_TRIAL_COUNT',
  'datetime_completed': None,
  'datetime_started': 'Thu, 06 Sep 2018 04:48:58 GMT',
  'id': 'd6a00b3a-317d-4804-ae1f-9a7d0e57d75f',
  'status': 'STARTED',
  'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
  'test_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=test',
  'train_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=train'}]
```

Viewing details of the latest train job of an app:

```py
client.get_train_job_of_app(app='fashion_mnist_app')
```

```sh
[{'app': 'fashion_mnist_app',
  'app_version': 1,
  'budget_amount': 10,
  'budget_type': 'MODEL_TRIAL_COUNT',
  'datetime_completed': None,
  'datetime_started': 'Thu, 06 Sep 2018 04:48:58 GMT',
  'id': 'd6a00b3a-317d-4804-ae1f-9a7d0e57d75f',
  'models': ['single_hidden_layer_tf'],
  'status': 'STARTED',
  'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
  'test_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=test',
  'train_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=train',
  'workers': [{'datetime_started': 'Thu, 06 Sep 2018 04:49:00 GMT',
               'model': 'single_hidden_layer_tf',
               'replicas': 2,
               'service_id': '23nszw0dgi956qbz29xfsiws0',
               'status': 'RUNNING'}]}]
```

Viewing best trials of an app:

```py
client.get_best_trials_of_app(app='fashion_mnist_app')
```

```sh
[{'datetime_started': 'Thu, 06 Sep 2018 04:50:55 GMT',
  'hyperparameters': {'batch_size': 128,
                      'epochs': 1,
                      'hidden_layer_units': 85,
                      'learning_rate': 0.003984521706625327},
  'id': '93aae9c7-e702-4e3b-8658-ff0dab5def6e',
  'model_name': 'single_hidden_layer_tf',
  'score': 0.8532,
  'train_job_id': 'd6a00b3a-317d-4804-ae1f-9a7d0e57d75f'},
 {'datetime_started': 'Thu, 06 Sep 2018 04:50:26 GMT',
  'hyperparameters': {'batch_size': 8,
                      'epochs': 1,
                      'hidden_layer_units': 84,
                      'learning_rate': 0.00042853939434250643},
  'id': 'de586911-9ace-4d5c-a3ca-2eeda740e195',
  'model_name': 'single_hidden_layer_tf',
  'score': 0.8449,
  'train_job_id': 'd6a00b3a-317d-4804-ae1f-9a7d0e57d75f'},
 {'datetime_started': 'Thu, 06 Sep 2018 04:50:15 GMT',
  'hyperparameters': {'batch_size': 32,
                      'epochs': 1,
                      'hidden_layer_units': 85,
                      'learning_rate': 0.00010346559524770284},
  'id': 'c2440e29-c7f8-42a4-ae48-abfa9cbb7cf3',
  'model_name': 'single_hidden_layer_tf',
  'score': 0.837,
  'train_job_id': 'd6a00b3a-317d-4804-ae1f-9a7d0e57d75f'}]
```

Viewing all trials of an app:

```py
client.get_trials_of_app(app='fashion_mnist_app')
```

Stopping a train job prematurely:

```py
client.stop_train_job(train_job_id=<train_job_id>)
```
