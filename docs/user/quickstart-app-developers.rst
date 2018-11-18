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

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client(admin_host='localhost', admin_port=3000)
        client.login(email='app_developer@rafiki', password='rafiki')
        
.. seealso:: :meth:`rafiki.client.Client.login`

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

Example:

    .. code-block:: python

        client.get_train_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'budget': {'MODEL_TRIAL_COUNT': 2},
        'datetime_completed': None,
        'datetime_started': 'Sun, 18 Nov 2018 09:56:36 GMT',
        'id': '3f3b3bdd-43ac-4354-99a5-d4d86006b68a',
        'status': 'RUNNING',
        'task': 'IMAGE_CLASSIFICATION',
        'test_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true',
        'train_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true',
        'workers': [{'datetime_started': 'Sun, 18 Nov 2018 09:56:36 GMT',
                    'datetime_stopped': None,
                    'model_name': 'TfFeedForward',
                    'replicas': 2,
                    'service_id': '64aeefae-eb49-416e-9488-4f22b19f55c7',
                    'status': 'RUNNING'},
                    {'datetime_started': 'Sun, 18 Nov 2018 09:56:36 GMT',
                    'datetime_stopped': None,
                    'model_name': 'SkDt',
                    'replicas': 2,
                    'service_id': '17adee63-3a8a-4e18-baa8-48f3c8d0af42',
                    'status': 'RUNNING'}]}

.. seealso:: :meth:`rafiki.client.Client.get_train_job`

Listing best trials of the latest train job for an app
--------------------------------------------------------------------

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

.. seealso:: :meth:`rafiki.client.Client.get_best_trials_of_train_job`

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
        'datetime_started': 'Sun, 18 Nov 2018 10:04:13 GMT',
        'datetime_stopped': None,
        'id': '9bcf0fb9-0bd5-4e76-a730-44b6d2370695',
        'predictor_host': '127.0.0.1:30000',
        'status': 'RUNNING',
        'train_job_id': '3f3b3bdd-43ac-4354-99a5-d4d86006b68a',
        'workers': [{'datetime_started': 'Sun, 18 Nov 2018 10:04:13 GMT',
                    'datetime_stopped': None,
                    'replicas': 2,
                    'service_id': '2ca0e607-9e1b-4292-b13d-08bc8d75e5b7',
                    'status': 'RUNNING',
                    'trial': {'id': '64984d19-ea18-4d4a-9cf9-9681ff829939',
                                'knobs': {'batch_size': 128,
                                        'epochs': 3,
                                        'hidden_layer_count': 2,
                                        'hidden_layer_units': 6,
                                        'image_size': 32,
                                        'learning_rate': 0.00377395877597336},
                                'model_name': 'TfFeedForward',
                                'score': 0.8242}},
                    {'datetime_started': 'Sun, 18 Nov 2018 10:04:13 GMT',
                    'datetime_stopped': None,
                    'replicas': 2,
                    'service_id': 'b70c903e-1533-48bf-91c4-92bc5aeada6d',
                    'status': 'RUNNING',
                    'trial': {'id': '4dd981fd-82fa-4db6-8278-687fc38e7af0',
                                'knobs': {'criterion': 'entropy', 'max_depth': 5},
                                'model_name': 'SkDt',
                                'score': 0.7048}}]}


Stopping a running inference job
--------------------------------------------------------------------

.. include:: ./client-stop-inference-job.include.rst