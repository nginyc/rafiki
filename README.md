# Rafiki

*Rafiki* is a distributed, scalable system that trains machine learning (ML) models and deploys trained models, built with ease-of-use in mind. To do so, it leverages on automated machine learning (AutoML).

Visit Rafiki's documentation at https://nginyc.github.io/rafiki2/docs/.

## Installation

Prerequisites: MacOS or Linux environment

1. Install Docker 18

2. Install Python 3.6

## Quickstart

1. Create the configuration file `./.env.sh` for Rafiki:

    ```sh
    bash scripts/create_env_file.sh
    ```

2. Setup Rafiki's complete stack with the init script:

    ```sh
    bash scripts/start.sh
    ```

3. To destroy Rafiki's complete stack:

    ```sh
    bash scripts/stop.sh
    ```

## Development

### Building Images Locally

The quickstart instructions pull pre-built [Rafiki's images](https://hub.docker.com/r/rafikiai/) from Docker Hub. To build Rafiki's images locally (e.g. to reflect latest code changes):

```sh
bash scripts/build_images.sh
```

> If you're testing latest code changes on multiple nodes, you'll need to build Rafiki's images on those nodes as well.

### Pushing Images to Docker Hub

To push the Rafiki's latest images to Docker Hub (e.g. to reflect the latest code changes):

```sh
bash scripts/push_images.sh
```

### Building Rafiki's Documentation

Rafiki uses [Sphinx documentation](http://www.sphinx-doc.org) and hosts the documentation with [Github Pages](https://pages.github.com/) on the [`/gh-pages` branch](https://github.com/nginyc/rafiki2/tree/gh-pages). Build & view Rafiki's Sphinx documentation on your machine with the following commands:

```shell
pip install sphinx sphinx_rtd_theme
sphinx-build -b html . docs
open docs/index.html
```

### Starting Parts of the Stack

The quickstart instructions set up a single node Docker Swarm on your machine. Separate shell scripts in the `./scripts/` folder configure and start parts of Rafiki's stack. Refer to the commands in
`./scripts/start.sh`.

### Reading Rafiki's logs

You can read logs of Rafiki Admin, Rafiki Advisor & Rafiki's services in the logs directory:

```sh
open /var/log/rafiki
```

### Connecting to Rafiki's DB

By default, you can connect to the PostgreSQL DB using a PostgreSQL client (e.g [Postico](https://eggerapps.at/postico/)) with these credentials:

```sh
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=rafiki
POSTGRES_DB=rafiki
POSTGRES_PASSWORD=rafiki
```

### Connecting to Rafiki's Cache

You can connect to Redis DB with *rebrow*:

```sh
source .env.sh
bash scripts/start_rebrow.sh
```

...with these credentials by default:

```sh
REDIS_HOST=rafiki_cache
REDIS_PORT=6379
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

While building Rafiki's images locally, if you encounter an error like "No space left on device", you might be running out of space allocated for Docker. Try removing all containers & images:

```sh
# Delete all containers
docker rm $(docker ps -a -q)
# Delete all images
docker rmi $(docker images -q)
```
