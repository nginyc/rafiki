# Guide for App Developers

## Installation

1. Install Python 3.6

2. Setup & configure Raifki Client by running:

```sh
pip install -r ./rafiki/client/requirements.txt
```

## Using Rafiki

Use the Rafiki Client Python module on the Python CLI:

Logging in:

```py
from rafiki.client import Client
client = Client()
client.login(email='app_developer@rafiki', password='rafiki')
```

```sh
{'user_id': 'eb273359-c74b-492b-80af-b9ea47ca959a', 'user_type': 'APP_DEVELOPER'}
```

### Train Jobs

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

Creating a train job for an app:

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
{'app': 'fashion_mnist_app', 'app_version': 1, 'id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}
```

Viewing train jobs of an app:

```py
client.get_train_jobs_of_app(app='fashion_mnist_app')
```

```sh
[{'app': 'fashion_mnist_app',
  'app_version': 1,
  'budget_amount': 3,
  'budget_type': 'MODEL_TRIAL_COUNT',
  'datetime_completed': None,
  'datetime_started': 'Mon, 17 Sep 2018 05:00:24 GMT',
  'id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58',
  'status': 'RUNNING',
  'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
  'test_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=test',
  'train_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=train'}]
```

Viewing details of the latest train job of an app:

```py
client.get_train_job(app='fashion_mnist_app')
```

```sh
[{'app': 'fashion_mnist_app',
  'app_version': 1,
  'budget_amount': 3,
  'budget_type': 'MODEL_TRIAL_COUNT',
  'datetime_completed': 'Mon, 17 Sep 2018 05:04:26 GMT',
  'datetime_started': 'Mon, 17 Sep 2018 05:00:24 GMT',
  'id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58',
  'status': 'COMPLETED',
  'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
  'test_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=test',
  'train_dataset_uri': 'tf-keras://fashion_mnist?train_or_test=train',
  'workers': [{'datetime_started': 'Mon, 17 Sep 2018 05:00:25 GMT',
               'datetime_stopped': 'Mon, 17 Sep 2018 05:04:26 GMT',
               'model_name': 'single_hidden_layer_tf',
               'replicas': 2,
               'service_id': '09c18fd9-ee7d-44bf-b658-fa9c6d0972a3',
               'status': 'STOPPED'}]}]
```

Viewing best trials of a train job:

```py
client.get_best_trials_of_train_job(app='fashion_mnist_app')
```

```sh
[{'datetime_completed': 'Mon, 17 Sep 2018 05:00:56 GMT',
  'datetime_started': 'Mon, 17 Sep 2018 05:00:35 GMT',
  'hyperparameters': {'batch_size': 32,
                      'epochs': 1,
                      'hidden_layer_units': 10,
                      'learning_rate': 0.0075360338999624086},
  'id': '8d40ad88-e5a1-4b16-b188-44be920b1683',
  'model_name': 'single_hidden_layer_tf',
  'score': 0.8231},
 {'datetime_completed': 'Mon, 17 Sep 2018 05:04:26 GMT',
  'datetime_started': 'Mon, 17 Sep 2018 05:03:06 GMT',
  'hyperparameters': {'batch_size': 1,
                      'epochs': 1,
                      'hidden_layer_units': 10,
                      'learning_rate': 0.030337360568713518},
  'id': '74bd9b43-9812-4930-a29c-9b765b5b46bc',
  'model_name': 'single_hidden_layer_tf',
  'score': 0.099},
 {'datetime_completed': 'Mon, 17 Sep 2018 05:03:06 GMT',
  'datetime_started': 'Mon, 17 Sep 2018 05:00:56 GMT',
  'hyperparameters': {'batch_size': 1,
                      'epochs': 1,
                      'hidden_layer_units': 78,
                      'learning_rate': 0.056356430854509774},
  'id': '94ea26de-e4a1-45af-8907-51cc4509d410',
  'model_name': 'single_hidden_layer_tf',
  'score': 0.092}]
```

Viewing all trials of a train job:

```py
client.get_trials_of_train_job(app='fashion_mnist_app')
```

Stopping the latest train job of an app prematurely:

```py
client.stop_train_job(app='fashion_mnist_app')
```

### Inference Jobs

Creating an inference job for an app:

```py
client.create_inference_job(app='fashion_mnist_app')
```

```sh
{'app': 'fashion_mnist_app', 'app_version': 1, 'id': '25c117a0-1677-44b2-affb-c56f8f99dabf', 'query_host': '192.168.1.75:30000', 'train_job_id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}
```

Viewing inference jobs of an app:

```py
client.get_inference_jobs_of_app(app='fashion_mnist_app')
```

```sh
[{'app': 'fashion_mnist_app',
  'app_version': 1,
  'datetime_started': 'Mon, 17 Sep 2018 05:17:34 GMT',
  'datetime_stopped': None,
  'id': '25c117a0-1677-44b2-affb-c56f8f99dabf',
  'query_host': '192.168.1.75:30000',
  'status': 'RUNNING',
  'train_job_id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}]
```

Viewing details of the latest inference job of an app:

```py
client.get_inference_job(app='fashion_mnist_app')
```

```sh
[{'app': 'fashion_mnist_app',
  'app_version': 1,
  'datetime_started': 'Mon, 17 Sep 2018 05:17:34 GMT',
  'datetime_stopped': None,
  'id': '25c117a0-1677-44b2-affb-c56f8f99dabf',
  'query_host': '192.168.1.75:30000',
  'status': 'RUNNING',
  'train_job_id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58',
  'workers': [{'datetime_started': 'Mon, 17 Sep 2018 05:17:34 GMT',
               'datetime_stopped': None,
               'replicas': 2,
               'service_id': '27d1986f-f96c-4ada-ae35-d6cd1d55f8ca',
               'status': 'RUNNING',
               'trial': {'hyperparameters': {'batch_size': 32,
                                             'epochs': 1,
                                             'hidden_layer_units': 10,
                                             'learning_rate': 0.0075360338999624086},
                         'id': '8d40ad88-e5a1-4b16-b188-44be920b1683',
                         'model_name': 'single_hidden_layer_tf',
                         'score': 0.8231}},
              {'datetime_started': 'Mon, 17 Sep 2018 05:17:34 GMT',
               'datetime_stopped': None,
               'replicas': 2,
               'service_id': '951b78c8-dbc3-470c-8d5d-55db11eca6b0',
               'status': 'RUNNING',
               'trial': {'hyperparameters': {'batch_size': 1,
                                             'epochs': 1,
                                             'hidden_layer_units': 10,
                                             'learning_rate': 0.030337360568713518},
                         'id': '74bd9b43-9812-4930-a29c-9b765b5b46bc',
                         'model_name': 'single_hidden_layer_tf',
                         'score': 0.099}}]}]
```

Stopping an inference job:

```py
client.stop_inference_job(app='fashion_mnist_app')
```