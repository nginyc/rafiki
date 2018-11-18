Example:

    .. code-block:: python

        client.get_train_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'app': 'fashion_mnist_app',
        'app_version': 1,
        'budget': {'MODEL_TRIAL_COUNT': 2},
        'datetime_completed': None,
        'datetime_started': 'Sun, 18 Nov 2018 09:56:36 GMT',
        'id': '3f3b3bdd-43ac-4354-99a5-d4d86006b68a',
        'status': 'RUNNING',
        'task': 'IMAGE_CLASSIFICATION',
        'test_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true',
        'train_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true'}]

.. seealso:: :meth:`rafiki.client.Client.get_train_jobs_of_app`