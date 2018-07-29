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

This example uses the ["Titanic" dataset on Kaggle](https://www.kaggle.com/c/titanic/), residing in `./data/kaggle-titanic/`.

Pre-process the dataset:

```shell
python ./data/kaggle-titanic/prepare.py
```

This generates a `train_final.csv` in the CSV format expected by ATM.

Create a datarun of 30 min on this dataset with:

```shell
curl -sS \
  -d '{"dataset_url":"data/kaggle-titanic/train_final.csv", "class_column":"Survived", "budget_type": "walltime", "budget": 30}' \
  -H "Content-Type: application/json" \
  -X POST \
  http://<docker_host>:8000/dataruns
```

View the properties of the datarun, including its `status` & latest `best_classifier_id`, with:

```shell
curl -sS http://<docker_host>:8000/dataruns/<datarun_id>
```

Retrieve the properties of this best classifier, including its `hyperparameters` and `cv_accuracy` with:

```shell
curl -sS http://<docker_host>:8000/classifiers/<best_classifier_id>
```

Make batch predictions on this best classifier with e.g.:

```shell
curl -sS \
  -d '{
    "queries": [
      {
        "Pclass": 0,
        "Sex-Male": 1,
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Cabin-A": 0,
        "Cabin-B": 0,
        "Cabin-C": 0,
        "Cabin-D": 0,
        "Cabin-E": 0,
        "Cabin-F": 0,
        "Cabin-G": 0,
        "Cabin_Count": 0,
        "Embarked-C": 0,
        "Embarked-Q": 0,
        "Embarked-S": 1
      },
      {
        "Pclass": 1,
        "Sex-Male": 0,
        "Age": 38,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.2833,
        "Cabin-A": 0,
        "Cabin-B": 0,
        "Cabin-C": 1,
        "Cabin-D": 0,
        "Cabin-E": 0,
        "Cabin-F": 0,
        "Cabin-G": 0,
        "Cabin_Count": 0,
        "Embarked-C": 1,
        "Embarked-Q": 0,
        "Embarked-S": 0
      }
    ]
  }' \
  -H "Content-Type: application/json" \
  -X POST \
  http://<docker_host>:8000/classifiers/<best_classifier_id>/queries
```

## TODO

- Add custom algorithms to ATM e.g. deep learning models
- Try images as input
- Find a more distributed way of sharing model data

## Resources

## Credits

Original Auto-Tune Models (ATM) project: https://github.com/HDI-Project/ATM