.. _`quickstart-app-developers`:

Quick Start (Application Developers)
====================================================================

.. contents:: Table of Contents

We assume that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
and *Rafiki Admin Web* at ``<rafiki_host>:<admin_web_port>``.

Installing the Client
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the Client
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.login`

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client(admin_host='localhost', admin_port=8000)
        client.login(email='superadmin@rafiki', password='rafiki')
        

Listing models by task
--------------------------------------------------------------------

.. include:: ./client-list-models.include.rst


Creating a train job
--------------------------------------------------------------------

.. include:: ./client-create-train-job.include.rst


Listing train jobs of an app
--------------------------------------------------------------------

.. include:: ./client-list-train-jobs.include.rst


Retrieving the latest train job's details for an app
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_train_job`

Example:

    .. code-block:: python

        client.get_train_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'budget_amount': 3,
        'budget_type': 'MODEL_TRIAL_COUNT',
        'datetime_completed': 'Thu, 04 Oct 2018 03:27:51 GMT',
        'datetime_started': 'Thu, 04 Oct 2018 03:25:06 GMT',
        'id': '65af28c7-e3ef-4fb0-af76-8b413d16ad76',
        'status': 'COMPLETED',
        'task': 'IMAGE_CLASSIFICATION',
        'test_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true',
        'train_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true',
        'workers': [{'datetime_started': 'Thu, 04 Oct 2018 03:25:06 GMT',
                    'datetime_stopped': 'Thu, 04 Oct 2018 03:27:15 GMT',
                    'model_name': 'TfSingleHiddenLayer',
                    'replicas': 2,
                    'service_id': '584ec59d-99ab-4e93-ab3a-af844d325a37',
                    'status': 'STOPPED'},
                    {'datetime_started': 'Thu, 04 Oct 2018 03:25:06 GMT',
                    'datetime_stopped': 'Thu, 04 Oct 2018 03:27:51 GMT',
                    'model_name': 'SkDt',
                    'replicas': 2,
                    'service_id': '6898ed0a-d39c-49fe-bbad-1ce5b01fd2dd',
                    'status': 'STOPPED'}]}


Listing best trials of the latest train job for an app
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_best_trials_of_train_job`

Example:

    .. code-block:: python

        client.get_best_trials_of_train_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'datetime_started': 'Thu, 04 Oct 2018 03:26:54 GMT',
        'datetime_stopped': 'Thu, 04 Oct 2018 03:27:04 GMT',
        'id': '38383e64-4406-4292-9e4b-abe342a085d3',
        'knobs': {'batch_size': 128,
                    'epochs': 1,
                    'hidden_layer_units': 34,
                    'learning_rate': 0.022193442791377953},
        'model_name': 'TfSingleHiddenLayer',
        'score': 0.8312},
        {'datetime_started': 'Thu, 04 Oct 2018 03:25:18 GMT',
        'datetime_stopped': 'Thu, 04 Oct 2018 03:27:39 GMT',
        'id': 'd8ea9d7f-c484-462b-80cb-dfa01f07d9c1',
        'knobs': {'criterion': 'entropy', 'max_depth': 8},
        'model_name': 'SkDt',
        'score': 0.7823},
        {'datetime_started': 'Thu, 04 Oct 2018 03:25:18 GMT',
        'datetime_stopped': 'Thu, 04 Oct 2018 03:27:23 GMT',
        'id': '99bc9628-be0d-406e-97f8-1566aa58ffd7',
        'knobs': {'criterion': 'entropy', 'max_depth': 6},
        'model_name': 'SkDt',
        'score': 0.7341}]

.. _`creating-inference-job`:

Creating an inference job with the latest train job for an app
--------------------------------------------------------------------

Your app's users will make queries to the `/predict` endpoint of *predictor_host* over HTTP.

.. seealso:: :ref:`making-predictions` 

.. include:: ./client-create-inference-job.include.rst
    

Listing inference jobs of an app
--------------------------------------------------------------------

.. include:: ./client-list-inference-jobs.include.rst


Retrieving details of running inference job for an app 
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_running_inference_job`

Example:

    .. code-block:: python

        client.get_running_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'datetime_started': 'Thu, 04 Oct 2018 03:31:59 GMT',
        'datetime_stopped': None,
        'id': '38c53776-c450-4b86-a173-6e245863549a',
        'predictor_host': '127.0.0.1:30000',
        'status': 'RUNNING',
        'train_job_id': '65af28c7-e3ef-4fb0-af76-8b413d16ad76',
        'workers': [{'datetime_started': 'Thu, 04 Oct 2018 03:31:59 GMT',
                    'datetime_stopped': None,
                    'replicas': 2,
                    'service_id': '13e21391-c054-489e-819b-90fd1ab175bb',
                    'status': 'RUNNING',
                    'trial': {'id': '38383e64-4406-4292-9e4b-abe342a085d3',
                                'knobs': {'batch_size': 128,
                                        'epochs': 1,
                                        'hidden_layer_units': 34,
                                        'learning_rate': 0.022193442791377953},
                                'model_name': 'TfSingleHiddenLayer',
                                'score': 0.8312}},
                    {'datetime_started': 'Thu, 04 Oct 2018 03:31:59 GMT',
                    'datetime_stopped': None,
                    'replicas': 2,
                    'service_id': 'f25fb364-270f-4b11-b5d5-4466dbfdfb0b',
                    'status': 'RUNNING',
                    'trial': {'id': 'd8ea9d7f-c484-462b-80cb-dfa01f07d9c1',
                                'knobs': {'criterion': 'entropy', 'max_depth': 8},
                                'model_name': 'SkDt',
                                'score': 0.7823}}]}


Stopping a running inference job
--------------------------------------------------------------------

.. include:: ./client-stop-inference-job.include.rst