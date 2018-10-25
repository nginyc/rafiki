.. seealso:: :meth:`rafiki.client.Client.create_train_job`

A train job is uniquely identified by its associated app and the app version (returned in output).

Example:

    .. code-block:: python

        client.create_train_job(
            app='fashion_mnist_app',
            task='IMAGE_CLASSIFICATION',
            train_dataset_uri='https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_train.zip?raw=true',
            test_dataset_uri='https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_test.zip?raw=true',
            budget_type='MODEL_TRIAL_COUNT',
            budget_amount=2
        )

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '65af28c7-e3ef-4fb0-af76-8b413d16ad76'}

