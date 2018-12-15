.. _`quickstart-app-developers`:

Quick Start (Application Developers)
====================================================================

.. contents:: Table of Contents

As an App Developer, you can manage train & inference jobs on Rafiki.

To learn more about what you can do on Rafiki, explore the methods of :class:`rafiki.client.Client`.

We assume that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
and *Rafiki Admin Web* at ``<rafiki_host>:<admin_web_port>``.

Installing the client
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the client
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


Listing train jobs
--------------------------------------------------------------------

.. include:: ./client-list-train-jobs.include.rst


Retrieving the latest train job's details
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

Listing best trials of the latest train job
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

Creating an inference job with the latest train job
--------------------------------------------------------------------

Your app's users will make queries to the `/predict` endpoint of *predictor_host* over HTTP.

.. seealso:: :ref:`making-predictions` 

.. include:: ./client-create-inference-job.include.rst
    

Listing inference jobs
--------------------------------------------------------------------

.. include:: ./client-list-inference-jobs.include.rst


Retrieving details of running inference job
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


Downloading the trained model for a trial
--------------------------------------------------------------------

After running a train job, you might want to download the trained model instance 
of a trial of the train job, instead of creating an inference job to make predictions.
Subsequently, you'll be able to make batch predictions locally with the trained model instance.

To do this, you must have the trial's model class file already in your local filesystem,
the dependencies of the model must have been installed separately, and the model class must have been 
imported and passed into this method.

To download the model class file, use the method :meth:`rafiki.client.Client.download_model_file`.

Example:

    In shell,

    .. code-block:: shell

            # Install the dependencies of the `SkDt` model
            pip install scikit-learn==0.20.0

    In Python,

    .. code-block:: python

            # Find the best trial for model `SkDt`
            trials = [x for x in client.get_best_trials_of_train_job(app='fashion_mnist_app') if x.get('model_name') == 'SkDt']
            trial = trials[0]
            trial_id = trial.get('id')
            
            # Import the model class
            from examples.models.image_classification.SkDt import SkDt

            # Load an instance of the model with trial's parameters
            model_inst = client.load_trial_model(trial_id, SkDt)

            # Make predictions with trained model instance associated with best trial
            queries = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], 
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], 
                        [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], 
                        [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], 
                        [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], 
                        [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], 
                        [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], 
                        [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], 
                        [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], 
                        [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
            print(model_inst.predict(queries))

.. seealso:: :meth:`rafiki.client.Client.load_trial_model`