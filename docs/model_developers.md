# Guide for Model Developers

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
client.login(email='model_developer@rafiki', password='rafiki')
```

```sh
{'user_id': 'a8959685-6667-41d5-8f91-b195fda27f91', 'user_type': 'MODEL_DEVELOPER'}
```

Creating models:

Most likely, you'll be using a deep learning framework e.g. Tensorflow to build your model. You might want to build a custom Docker image for the model training & inference. This Docker image has to extend `rafiki_model`.

```sh
pip install tensorflow==1.10.1 h5py==2.8.0
bash src/model/scripts/build_tensorflow_worker_image.sh
```

```python
from model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel
model_inst = SingleHiddenLayerTensorflowModel()
client.create_model(
    name='single_hidden_layer_tf',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    model_inst=model_inst,
    docker_image='rafiki_tf_model'
)
```

```sh
{'name': 'single_hidden_layer_tf'}
```

Viewing models:

```py
client.get_models()
```

```sh
[{'datetime_created': 'Thu, 06 Sep 2018 04:38:48 GMT',
  'docker_image': 'rafiki_model',
  'name': 'single_hidden_layer_tf',
  'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
  'user_id': 'a8959685-6667-41d5-8f91-b195fda27f91'}]
```