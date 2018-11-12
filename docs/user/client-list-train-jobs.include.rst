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
        'datetime_completed': 'Thu, 04 Oct 2018 03:27:51 GMT',
        'datetime_started': 'Thu, 04 Oct 2018 03:25:06 GMT',
        'id': '65af28c7-e3ef-4fb0-af76-8b413d16ad76',
        'status': 'COMPLETED',
        'task': 'IMAGE_CLASSIFICATION',
        'test_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true',
        'train_dataset_uri': 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true'}]
