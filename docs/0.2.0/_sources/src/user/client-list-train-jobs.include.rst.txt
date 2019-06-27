Example:

    .. code-block:: python

        client.get_train_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'app': 'fashion_mnist_app',
        'app_version': 1,
        'budget': {'MODEL_TRIAL_COUNT': 5},
        'datetime_started': 'Mon, 17 Dec 2018 07:08:05 GMT',
        'datetime_stopped': None,
        'id': 'ec4db479-b9b2-4289-8086-52794ffc71c8',
        'status': 'RUNNING',
        'task': 'IMAGE_CLASSIFICATION',
        'val_dataset_id': '7e9a2f8a-c61d-4365-ae4a-601e90892b88',
        'train_dataset_id': 'ecf87d2f-6893-4e4b-8ed9-1d9454af9763'}]

.. seealso:: :meth:`rafiki.client.Client.get_train_jobs_of_app`