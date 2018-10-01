.. _`quickstart-app-developers`:

Quickstart (Application Developers)
====================================================================

.. contents:: Table of Contents

This quickstart submits the `Fashion MNIST dataset <https://github.com/zalandoresearch/fashion-mnist>`_ for training and inference.

Installation
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the Client
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.login`

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client()
        client.login(email='app_developer@rafiki', password='rafiki')
        

Listing models by task
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_models_of_task`

Example:

    .. code-block:: python

        client.get_models_of_task(task='IMAGE_CLASSIFICATION_WITH_ARRAYS')

    Output:

    .. code-block:: python

        [{'datetime_created': 'Tue, 25 Sep 2018 09:28:44 GMT',
        'docker_image': 'rafikiai/rafiki_worker',
        'model_class': 'SingleHiddenLayerTensorflowModel',
        'name': 'single_hidden_layer_tf',
        'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
        'user_id': '695a23b0-a488-4110-aa50-6476e53a0e4d'}]
    

Creating a train job
--------------------------------------------------------------------

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

.. _`creating-inference-job`:

Creating an inference job with the latest train job for an app
--------------------------------------------------------------------

An inference job is created from the trials of an associated train job, 
and uniquely identified by that train job's associated app and the app version.

Your app's users will make queries to the `/predict` endpoint of *predictor_host* over HTTP.

.. seealso:: :meth:`rafiki.client.Client.create_inference_job`
.. seealso:: :ref:`making-predictions` 

Example:

    .. code-block:: python

        client.create_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '25c117a0-1677-44b2-affb-c56f8f99dabf',
        'predictor_host': '127.0.0.1:30000',
        'train_job_id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}
    

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
        'predictor_host': '127.0.0.1:30000',
        'status': 'RUNNING',
        'train_job_id': '99b6a250-d0d0-431f-8fa7-eeedcd9bed58'}]


Retrieving details of running inference job for an app 
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_running_inference_job`

Example:

    .. code-block:: python

        client.get_running_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'app': 'fashion_mnist_app',
        'app_version': 1,
        'datetime_started': 'Mon, 17 Sep 2018 05:17:34 GMT',
        'datetime_stopped': None,
        'id': '25c117a0-1677-44b2-affb-c56f8f99dabf',
        'predictor_host': '127.0.0.1:30000',
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

.. seealso:: :meth:`rafiki.client.Client.stop_inference_job`

Example:

    .. code-block:: python

        client.stop_inference_job(app='fashion_mnist_app')
