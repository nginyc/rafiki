# Guide for Rafiki Admins

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
client.login(email='superadmin@rafiki', password='rafiki')
```

```py
{'user_id': '36641447-1105-4fd4-9797-377bd71cf561', 'user_type': 'SUPERADMIN'}
```

Creating users:

```py
client.create_user(
    email='admin@rafiki',
    password='rafiki',
    user_type='ADMIN'
)
```

```py
client.create_user(
    email='app_developer@rafiki',
    password='rafiki',
    user_type='APP_DEVELOPER'
)
```

```py
client.create_user(
    email='model_developer@rafiki',
    password='rafiki',
    user_type='MODEL_DEVELOPER'
)
```