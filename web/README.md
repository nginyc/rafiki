<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

# Rafiki Dashboard

## How to run the app

```
yarn start
```

## How to setup Api End Point (environment parameters)

```
rafiki $ source env.sh
```


```
rafiki/web $ vim .env
```

```
PORT=$WEB_ADMIN_EXT_PORT
NODE_PATH=./src
REACT_APP_API_POINT_HOST=$RAFIKI_ADDR
REACT_APP_API_POINT_PORT=$ADMIN_EXT_PORT
```

## How to TEST the app

```
yarn test
```

https://github.com/facebook/jest/issues/3254

If you cannot test on Ubuntu, maybe it is because jest is watching too many files.

```
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
```

### migrate from `react-testing-library` to `@testing-library/react`

```
react-testing-library has moved to @testing-library/react. Please uninstall react-testing-library and install @testing-library/react instead, or use an older version of react-testing-library. Learn more about this change here: https://github.com/testing-library/dom-testing-library/issues/260 Thanks! :)
```

## API Endpoints

## users

### post `/token`

header: content_type = 'application/json'

#### request
``` json
{
	"email": "superadmin@rafiki",
	"password": "rafiki"
}
```

#### response

``` json
{
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiYWUxN2FhOWYtNWRmNi00MWVjLTgzNDEtNWM2N2NjYmU0NTAxIiwidXNlcl90eXBlIjoiU1VQRVJBRE1JTiJ9.dxWzHxJ2nz_cX0oeiyOQY3wWD4irlpT47e-usfrVSWw",
    "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501",
    "user_type": "SUPERADMIN"
}
```


## Datasets

### get `/datasets`

```

```

### psot `datasets`

```

```

## models

### post `models` (working)

should use muitlform but does not need to set it because the browser would set it automatically.

### get `models`

#### header 

```
    'Authorization': `Bearer ${this._token}`
```

#### response

``` json
[
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:05:56 GMT",
        "dependencies": {
            "tensorflow": "1.12.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "TfFeedForward",
        "name": "TfFeedForward_GOZ9JANFQVVQT6LX",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:05:56 GMT",
        "dependencies": {
            "scikit-learn": "0.20.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "SkDt",
        "name": "SkDt_GOZ9JANFQVVQT6LX",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:10:27 GMT",
        "dependencies": {
            "tensorflow": "1.12.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "TfFeedForward",
        "name": "TfFeedForward_RY0DI2PV26UMPJ2L",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:10:27 GMT",
        "dependencies": {
            "scikit-learn": "0.20.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "SkDt",
        "name": "SkDt_RY0DI2PV26UMPJ2L",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Thu, 30 May 2019 06:07:09 GMT",
        "dependencies": {
            "tensorflow": "1.12.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "TfFeedForward",
        "name": "TfFeedForward",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    }
]
```

## trainJobs

## post `/train_job`

### requrest

#### header 
```
     'Authorization': `Bearer ${this._token}`
     'Content-type': 'application/json'
```
#### body

```
{
    "app":"fashion_mnist_app",
    "task": "IMAGE_CLASSIFICATION",
    "train_dataset_uri":"https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true",
    "test_dataset_uri":"https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true",
        "budget": { "MODEL_TRIAL_COUNT": 2 }
    "models" ["TfFeedForward"]
}
```

### response

```
{
    "app": "fashion_mnist_app",
    "app_version": 1,
    "id": "df266b4c-5a04-43f8-a751-230b5c7fec8f"
}
```

### get `/train_jobs?user_id={user_id}`

#### header

```
    'Authorization': `Bearer ${this._token}`
```

#### response

```
[
    {
        "app": "image_classification_app_5CWOI56WR7ABMKJV",
        "app_version": 1,
        "budget": {
            "ENABLE_GPU": 0,
            "MODEL_TRIAL_COUNT": 2
        },
        "datetime_started": "Tue, 26 Mar 2019 04:29:30 GMT",
        "datetime_stopped": "Tue, 26 Mar 2019 04:30:40 GMT",
        "id": "db738d0e-9a5f-48af-bde1-28b1c71bee02",
        "status": "STOPPED",
        "task": "IMAGE_CLASSIFICATION",
        "test_dataset_uri": "https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true",
        "train_dataset_uri": "https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true"
    },
	......
]
```

### post `/inference_jobs`

#### request 

```
{ 
  'user_id': user_id
}
```

#### response

```
{

}
```

#### NOTES:

Not considering moving this project to Typescript, because compile time is quite long.

https://stackoverflow.com/questions/47508564/migrating-create-react-app-from-javascript-to-typescript

#### TODO

- enabling the ui to add new application page as plugin (by 15.June), this needs to refractor the code.

- uploading data(both traning data & application data) (by 30.May)

- sync data/application status with backend (by 30.May)

- fixed trial page (by 30.May)

- using redux to handle the data (by 7.June)

