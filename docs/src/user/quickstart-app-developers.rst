.. _`quickstart-app-developers`:

Quick Start (Application Developers)
====================================================================

As an *App Developer*, you can manage datasets, train jobs & inference jobs on Rafiki. This guide walks through a *full* train-inference flow:

    1. Authenticating on Rafiki
    2. Uploading datasets
    3. Creating a model training job
    4. Creating a model serving job after the model training job completes
    
This guide assumes that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
and *Rafiki Web Admin* at ``<rafiki_host>:<web_admin_port>``, and there have been models added to Rafiki under the task of `IMAGE_CLASSIFICATION`.

To learn more about what else you can do on Rafiki, explore the methods of :class:`rafiki.client.Client`.

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

Listing available models by task
--------------------------------------------------------------------

.. include:: ./client-list-models.include.rst


Creating datasets
--------------------------------------------------------------------

.. include:: ./client-create-datasets.include.rst


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
        'datetime_started': 'Mon, 17 Dec 2018 07:08:05 GMT',
        'datetime_stopped': 'Mon, 17 Dec 2018 07:11:11 GMT',
        'id': 'ec4db479-b9b2-4289-8086-52794ffc71c8',
        'status': 'STOPPED',
        'task': 'IMAGE_CLASSIFICATION'
        'val_dataset_id': '7e9a2f8a-c61d-4365-ae4a-601e90892b88',
        'train_dataset_id': 'ecf87d2f-6893-4e4b-8ed9-1d9454af9763',
        'workers': [{'datetime_started': 'Mon, 17 Dec 2018 07:08:05 GMT',
                    'datetime_stopped': 'Mon, 17 Dec 2018 07:11:14 GMT',
                    'model_name': 'SkDt',
                    'replicas': 2,
                    'service_id': '2ada1ff3-84e9-4eca-bac9-241cd8c765ef',
                    'status': 'STOPPED'},
                    {'datetime_started': 'Mon, 17 Dec 2018 07:08:05 GMT',
                    'datetime_stopped': 'Mon, 17 Dec 2018 07:11:42 GMT',
                    'model_name': 'TfFeedForward',
                    'replicas': 2,
                    'service_id': '81ff23a7-ddd0-4a62-9d86-a3cc985ca6fe',
                    'status': 'STOPPED'}]}

.. seealso:: :meth:`rafiki.client.Client.get_train_job`

Listing best trials of the latest train job
--------------------------------------------------------------------

Example:

    .. code-block:: python

        client.get_best_trials_of_train_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'datetime_started': 'Mon, 17 Dec 2018 07:09:17 GMT',
        'datetime_stopped': 'Mon, 17 Dec 2018 07:11:38 GMT',
        'id': '1b7dc65a-87ae-4d42-9a01-67602115a4a4',
        'knobs': {'batch_size': 32,
                    'epochs': 3,
                    'hidden_layer_count': 2,
                    'hidden_layer_units': 36,
                    'image_size': 32,
                    'learning_rate': 0.014650971133579896},
        'model_name': 'TfFeedForward',
        'score': 0.8269},
        {'datetime_started': 'Mon, 17 Dec 2018 07:08:38 GMT',
        'datetime_stopped': 'Mon, 17 Dec 2018 07:11:11 GMT',
        'id': '0c1f9184-7b46-4aaf-a581-be62bf3f49bf',
        'knobs': {'criterion': 'entropy', 'max_depth': 4},
        'model_name': 'SkDt',
        'score': 0.6686}]

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
        'datetime_started': 'Mon, 17 Dec 2018 07:25:36 GMT',
        'datetime_stopped': None,
        'id': '09e5040e-2134-411b-855f-793927c80b4b',
        'predictor_host': '127.0.0.1:30000',
        'status': 'RUNNING',
        'train_job_id': 'ec4db479-b9b2-4289-8086-52794ffc71c8',
        'workers': [{'datetime_started': 'Mon, 17 Dec 2018 07:25:36 GMT',
                    'datetime_stopped': None,
                    'replicas': 2,
                    'service_id': '661035bb-3966-46e8-828c-e200960a76c0',
                    'status': 'RUNNING',
                    'trial': {'id': '1b7dc65a-87ae-4d42-9a01-67602115a4a4',
                                'knobs': {'batch_size': 32,
                                        'epochs': 3,
                                        'hidden_layer_count': 2,
                                        'hidden_layer_units': 36,
                                        'image_size': 32,
                                        'learning_rate': 0.014650971133579896},
                                'model_name': 'TfFeedForward',
                                'score': 0.8269}},
                    {'datetime_started': 'Mon, 17 Dec 2018 07:25:36 GMT',
                    'datetime_stopped': None,
                    'replicas': 2,
                    'service_id': '6a769007-b18f-4271-b3db-8b60ed5fb545',
                    'status': 'RUNNING',
                    'trial': {'id': '0c1f9184-7b46-4aaf-a581-be62bf3f49bf',
                                'knobs': {'criterion': 'entropy', 'max_depth': 4},
                                'model_name': 'SkDt',
                                'score': 0.6686}}]}


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

            # Install the dependencies of the `TfFeedForward` model
            pip install tensorflow==1.12.0

    In Python,

    .. code-block:: python

            # Find the best trial for model `TfFeedForward`
            trials = [x for x in client.get_best_trials_of_train_job(app='fashion_mnist_app') 
                if x.get('model_name') == 'TfFeedForward' and x.get('status') == 'COMPLETED']
            trial = trials[0]
            trial_id = trial.get('id')
            
            # Import the model class
            from examples.models.image_classification.TfFeedForward import TfFeedForward

            # Load an instance of the model with trial's parameters
            model_inst = client.load_trial_model(trial_id, TfFeedForward)

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