.. seealso:: :meth:`rafiki.client.Client.create_train_job`

A train job is uniquely identified by its associated app and the app version (returned in output).

To create a train job, you'll need to prepare your dataset in a format specified by the target task, 
and upload it to a publicly accessible URL. 

.. seealso:: :ref:`tasks`

You can monitor your train jobs on Rafiki's Admin Web's GUI, including plots of metrics for models that have been trained on your dataset.

Example:

    .. code-block:: python

        client.create_train_job(
            app='fashion_mnist_app',
            task='IMAGE_CLASSIFICATION',
            train_dataset_uri='https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true',
            test_dataset_uri='https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true',
            budget_type='MODEL_TRIAL_COUNT',
            budget_amount=2
        )

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '65af28c7-e3ef-4fb0-af76-8b413d16ad76'}

.. note::

    The datasets in the above example have been pre-processed to conform to the task's dataset specification :ref:`dataset-type:IMAGE_FILES`. 
    The code that does this pre-processing from the original Fashion MNIST dataset is available at `./examples/datasets/image_files/load_mnist_format.py <https://github.com/nginyc/rafiki/tree/master/examples/datasets/image_files/load_mnist_format.py>`_.
    
