# Guide for Rafiki Admins

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
client.login(email='admin@rafiki', password='rafiki')
```

```py
{'user_id': 'e6078ff2-2257-4147-85a6-14a2d7730a8b', 'user_type': 'ADMIN'}
```

Creating users:

```py
client.create_user(
    email='app_developer@rafiki',
    password='rafiki',
    user_type='APP_DEVELOPER'
)
```

```sh
{'id': 'eb273359-c74b-492b-80af-b9ea47ca959a'}
```

```py
client.create_user(
    email='model_developer@rafiki',
    password='rafiki',
    user_type='MODEL_DEVELOPER'
)
```

```sh
{'id': 'a8959685-6667-41d5-8f91-b195fda27f91'}
```