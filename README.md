# Auto Tune Models Example

## Installation

1. Install Docker

2. Install Python 3.6 & install the project's Python dependencies by running `pip install -r ./requirements.txt`.

## Running the Stack

Create a .env at root of project:
```
MYSQL_HOST=<docker_host>
MYSQL_PORT=3306
MYSQL_USER=atm
MYSQL_DATABASE=atm
MYSQL_PASSWORD=atm
```

Run in terminal 1:

```shell
bash scripts/start_db.sh
```

Run in terminal 2:

```shell
bash scripts/start_admin.sh
```

Run in terminal 3:

```shell
bash scripts/start_worker.sh
```

This example uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

Create a datarun of 100 classifiers on this dataset with:

```shell
curl -sS \
  -d '{
    "dataset_name": "fashion_mnist",
    "preparator_type": "tf_keras_dataset",
    "preparator_params": {
      "keras_dataset_name": "fashion_mnist"
    },
    "budget_type": "classifier",
    "budget": 100
  }' \
  -H "Content-Type: application/json" \
  -X POST \
  http://<docker_host>:8000/dataruns
```

View the properties of the datarun, including its `status` & latest `best_classifier_id`, with:

```shell
curl -sS http://<docker_host>:8000/dataruns/<datarun_id>
```

View the properties of a dataset & its examples with:

```shell
curl -sS http://<docker_host>:8000/datasets/<dataset_id>
curl -sS http://<docker_host>:8000/datasets/<dataset_id>/<example_id>
curl -sS http://<docker_host>:8000/datasets/<dataset_id>/random
```

View the properties of this best classifier, including its `hyperparameters` and `cv_accuracy` with:

```shell
curl -sS http://<docker_host>:8000/classifiers/<best_classifier_id>
```

Make batch predictions on this best classifier with e.g.:

```shell
curl -sS \
  -d '{
    {
    "queries": [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,13,73,0,0,1,4,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,3,0,36,136,127,62,54,0,0,0,1,3,4,0,0,3],[0,0,0,0,0,0,0,0,0,0,0,0,6,0,102,204,176,134,144,123,23,0,0,0,0,12,10,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,155,236,207,178,107,156,161,109,64,23,77,130,72,15],[0,0,0,0,0,0,0,0,0,0,0,1,0,69,207,223,218,216,216,163,127,121,122,146,141,88,172,66],[0,0,0,0,0,0,0,0,0,1,1,1,0,200,232,232,233,229,223,223,215,213,164,127,123,196,229,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,183,225,216,223,228,235,227,224,222,224,221,223,245,173,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,193,228,218,213,198,180,212,210,211,213,223,220,243,202,0],[0,0,0,0,0,0,0,0,0,1,3,0,12,219,220,212,218,192,169,227,208,218,224,212,226,197,209,52],[0,0,0,0,0,0,0,0,0,0,6,0,99,244,222,220,218,203,198,221,215,213,222,220,245,119,167,56],[0,0,0,0,0,0,0,0,0,4,0,0,55,236,228,230,228,240,232,213,218,223,234,217,217,209,92,0],[0,0,1,4,6,7,2,0,0,0,0,0,237,226,217,223,222,219,222,221,216,223,229,215,218,255,77,0],[0,3,0,0,0,0,0,0,0,62,145,204,228,207,213,221,218,208,211,218,224,223,219,215,224,244,159,0],[0,0,0,0,18,44,82,107,189,228,220,222,217,226,200,205,211,230,224,234,176,188,250,248,233,238,215,0],[0,57,187,208,224,221,224,208,204,214,208,209,200,159,245,193,206,223,255,255,221,234,221,211,220,232,246,0],[3,202,228,224,221,211,211,214,205,205,205,220,240,80,150,255,229,221,188,154,191,210,204,209,222,228,225,0],[98,233,198,210,222,229,229,234,249,220,194,215,217,241,65,73,106,117,168,219,221,215,217,223,223,224,229,29],[75,204,212,204,193,205,211,225,216,185,197,206,198,213,240,195,227,245,239,223,218,212,209,222,220,221,230,67],[48,203,183,194,213,197,185,190,194,192,202,214,219,221,220,236,225,216,199,206,186,181,177,172,181,205,206,115],[0,122,219,193,179,171,183,196,204,210,213,207,211,210,200,196,194,191,195,191,198,192,176,156,167,177,210,92],[0,0,74,189,212,191,175,172,175,181,185,188,189,188,193,198,204,209,210,210,211,188,188,194,192,216,170,0],[2,0,0,0,66,200,222,237,239,242,246,243,244,221,220,193,191,179,182,182,181,176,166,168,99,58,0,0],[0,0,0,0,0,0,0,40,61,44,72,41,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
}
  }' \
  -H "Content-Type: application/json" \
  -X POST \
  http://<docker_host>:8000/classifiers/<best_classifier_id>/queries
```

## Credits

Original Auto-Tune Models (ATM) project: https://github.com/HDI-Project/ATM