# Rafiki

## Installation

Prerequisites: MacOS or Linux environment

1. Install Docker 18

2. Install Python 3.6

## Quickstart

1. Create the configuration file `./.env.sh` for Rafiki:

    ```sh
    bash scripts/create_env_file.sh
    ```

2. Source the created `./.env.sh`:

    ```sh
    source ./.env.sh
    ```

3. Setup Rafiki's complete stack with the init script:

    ```sh
    bash scripts/start.sh
    ```

4. To destroy Rafiki's complete stack:

    ```sh
    bash scripts/stop.sh
    ```
    
        
If you are using multiple nodes, build these images on other nodes:

    ```sh
    bash scripts/build_image_model.sh
    bash scripts/build_query_frontend.sh
    ```


## Manual Setup

Shell scripts in the `./scripts/` folder build & run parts of Rafiki's stack. Refer to the commands in `./scripts/start.sh`.

## Using Rafiki

Visit Rafiki's documentation at https://nginyc.github.io/rafiki2/docs/.

## Building Rafiki's Documentation

Rafiki uses [Sphinx documentation](http://www.sphinx-doc.org) and hosts the documentation with [Github Pages](https://pages.github.com/) on the [`/gh-pages` branch](https://github.com/nginyc/rafiki2/tree/gh-pages). Build & view Rafiki's Sphinx documentation on your machine with the following commands:

```shell
pip install sphinx
sphinx-build -b html . docs
open docs/index.html
```

## Rafiki Admin HTTP Server REST API

To make calls to the HTTP endpoints, you'll need first authenticate with email & password against the `POST /tokens` endpoint to obtain an authentication token `token`, and subsequently add the `Authorization` header for every other call:

`Authorization: Bearer {{token}}`

The list of available HTTP endpoints & their request formats are available as a *Postman* collection (OUTDATED) in the root of this project.

### Creating a Model

For the `POST /models` endpoint, you'll need to first serialize the model:

```py
from rafiki.model import serialize_model_to_file
from rafiki.model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel
model_inst = SingleHiddenLayerTensorflowModel()
serialize_model_to_file(model_inst, out_file_path='model.pickle')
```

Then, together with the `name` & `task` fields, upload the output serialized model file as the `model_serialized` field of a multi-part form data request.

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

Next, you can connect to Redis with *rebrow*:

```sh
source .env.sh
bash scripts/start_rebrow.sh
```

...with these credentials by default:

```sh
REDIS_HOST=rafiki_cache
REDIS_PORT=6379
```

When running the whole stack locally, if you encounter an error like "No space left on device", you might be running out of space allocated for Docker. Try removing all containers & images:

```sh
# Delete all containers
docker rm $(docker ps -a -q)
# Delete all images
docker rmi $(docker images -q)
```
