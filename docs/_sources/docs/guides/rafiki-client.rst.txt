.. _rafiki-client:

Rafiki Client Quickstart
====================================================================

.. contents:: Table of Contents

Installation
--------------------------------------------------------------------

1. Install Python 3.6

2. Install Raifki Client's Python dependencies by running:
    .. code-block:: shell

        pip install -r ./rafiki/client/requirements.txt


Initializing the Client
--------------------------------------------------------------------

.. seealso:: :class:`rafiki.client.Client`

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client()


Logging in
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.login`

Example:

    .. code-block:: python
        
        client.login(email='superadmin@rafiki', password='rafiki')
        
Creating users
--------------------------------------------------------------------

Only admins can create users.

.. seealso:: :meth:`rafiki.client.Client.create_user`

Examples:

    .. code-block:: python

        client.create_user(
            email='app_developer@rafiki',
            password='rafiki',
            user_type='APP_DEVELOPER'
        )

    .. code-block:: python

        client.create_user(
            email='model_developer@rafiki',
            password='rafiki',
            user_type='MODEL_DEVELOPER'
        )


Creating models
--------------------------------------------------------------------

Only admins & model developers can create models.

Most likely, you'll be using a deep learning framework e.g. Tensorflow to build your model. 
The base Rafiki worker image has the following Python libraries pre-installed:

.. code-block:: text

    tensorflow==1.10.1
    h5py==2.8.0

You can optionally build a custom Docker image for the model training & inference and pass the argument for `docker_image`. 
This Docker image has to extend `rafiki_model`. An example is available at `./rafiki/model/dockerfiles/TensorflowModel.Dockerfile`.

.. seealso:: :meth:`rafiki.client.Client.create_model`

Example:

    Run the following to install Tensorflow:

    .. code-block:: shell

        pip install tensorflow==1.10.1 h5py==2.8.0


    Then, import the sample Tensorflow model and pass in an instance of the model:

    .. code-block:: python

        from rafiki.model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel
        model_inst = SingleHiddenLayerTensorflowModel()
        client.create_model(
            name='single_hidden_layer_tf',
            task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
            model_inst=model_inst
        )

Listing models by task
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_models_of_task`


Example:

    .. code-block:: python

        client.get_models_of_task(task='IMAGE_CLASSIFICATION_WITH_ARRAYS')

    Output:

    .. code-block:: python

        [{'datetime_created': 'Thu, 06 Sep 2018 04:38:48 GMT',
        'docker_image': 'rafiki_model',
        'name': 'single_hidden_layer_tf',
        'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
        'user_id': 'a8959685-6667-41d5-8f91-b195fda27f91'}]
    

Creating a train job
--------------------------------------------------------------------

Only admins & app developers can create train jobs.
A train job is uniquely identified by its associated app and the app version (returned in output).

.. seealso:: :meth:`rafiki.client.Client.create_train_job`

Example:

    .. code-block:: python

        client.create_train_job(
            app='fashion_mnist_app',
            task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
            train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
            test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test',
            budget_type='MODEL_TRIAL_COUNT',
            budget_amount=3
        )

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}


Listing train jobs of an app
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_train_jobs_of_app`

Example:

    .. code-block:: python

        client.get_train_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

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


Retrieving the latest train job's details for an app
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_train_job`

Example:

    .. code-block:: python

        client.get_train_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

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


Listing best trials of the latest train job for an app
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_best_trials_of_train_job`

Example:

    .. code-block:: python

        client.get_best_trials_of_train_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'datetime_stopped': 'Mon, 17 Sep 2018 05:00:56 GMT',
        'datetime_started': 'Mon, 17 Sep 2018 05:00:35 GMT',
        'knobs': {'batch_size': 32,
                            'epochs': 1,
                            'hidden_layer_units': 10,
                            'learning_rate': 0.0075360338999624086},
        'id': '8d40ad88-e5a1-4b16-b188-44be920b1683',
        'model_name': 'single_hidden_layer_tf',
        'score': 0.8231},
        {'datetime_stopped': 'Mon, 17 Sep 2018 05:04:26 GMT',
        'datetime_started': 'Mon, 17 Sep 2018 05:03:06 GMT',
        'knobs': {'batch_size': 1,
                            'epochs': 1,
                            'hidden_layer_units': 10,
                            'learning_rate': 0.030337360568713518},
        'id': '74bd9b43-9812-4930-a29c-9b765b5b46bc',
        'model_name': 'single_hidden_layer_tf',
        'score': 0.099},
        {'datetime_stopped': 'Mon, 17 Sep 2018 05:03:06 GMT',
        'datetime_started': 'Mon, 17 Sep 2018 05:00:56 GMT',
        'knobs': {'batch_size': 1,
                            'epochs': 1,
                            'hidden_layer_units': 78,
                            'learning_rate': 0.056356430854509774},
        'id': '94ea26de-e4a1-45af-8907-51cc4509d410',
        'model_name': 'single_hidden_layer_tf',
        'score': 0.092}]

Creating an inference job with the latest train job for an app
--------------------------------------------------------------------

Only admins & app developers can create inference jobs.
An inference job is created from the trials of an associated train job,
and uniquely identified by that train job's associated app and the app version.

.. seealso:: :meth:`rafiki.client.Client.create_inference_job`

Example:

    .. code-block:: python

        client.create_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '25c117a0-1677-44b2-affb-c56f8f99dabf',
        'query_host': '192.168.1.75:30000',
        'train_job_id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}
    

Making predictions with a running inference job
--------------------------------------------------------------------

Example:

    ``POST /predict`` to the inference job's query frontend at *query_host* 192.168.1.75:30000. E.g. in shell,

    .. code-block:: shell

        body='{"query": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}'
        curl -H "Content-Type: application/json" -X POST -d "$body" 192.168.1.75:30000/predict

    Output:

    .. code-block:: shell

        {
            "responses": [
                9,
                9
            ]
        }

Listing inference jobs of an app
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_inference_jobs_of_app`

Example:

    .. code-block:: python

        client.get_inference_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'app': 'fashion_mnist_app',
        'app_version': 1,
        'datetime_started': 'Mon, 17 Sep 2018 05:17:34 GMT',
        'datetime_stopped': None,
        'id': '25c117a0-1677-44b2-affb-c56f8f99dabf',
        'query_host': '192.168.1.75:30000',
        'status': 'RUNNING',
        'train_job_id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}]


Retrieving the latest inference job's details for an app
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_inference_job`

Example:

    .. code-block:: python

        client.get_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

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
                    'trial': {'knobs': {'batch_size': 32,
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
                    'trial': {'knobs': {'batch_size': 1,
                                                    'epochs': 1,
                                                    'hidden_layer_units': 10,
                                                    'learning_rate': 0.030337360568713518},
                                'id': '74bd9b43-9812-4930-a29c-9b765b5b46bc',
                                'model_name': 'single_hidden_layer_tf',
                                'score': 0.099}}]}]


Stopping a running inference job
--------------------------------------------------------------------

Only admins & app developers can stop inference jobs.

.. seealso:: :meth:`rafiki.client.Client.stop_inference_job`

Example:

    .. code-block:: python

        client.stop_inference_job(app='fashion_mnist_app')
