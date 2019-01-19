Example:

    .. code-block:: python

        client.get_train_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'app': 'fashion_mnist_app',
        'app_version': 1,
        'budget': {'MODEL_TRIAL_COUNT': 2},
        'datetime_started': 'Mon, 17 Dec 2018 07:08:05 GMT',
        'datetime_stopped': None,
        'id': 'ec4db479-b9b2-4289-8086-52794ffc71c8',
        'status': 'RUNNING',
        'task': 'IMAGE_CLASSIFICATION',
        'val_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_val.zip?raw=true',
        'train_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true'}]

.. seealso:: :meth:`rafiki.client.Client.get_train_jobs_of_app`