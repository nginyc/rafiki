.. seealso:: :meth:`rafiki.client.Client.create_inference_job`

An inference job is created from the trials of an associated train job, and the train job must have the status of ``COMPLETED``.
It is uniquely identified by that train job's associated app and the app version.

Example:

    .. code-block:: python

        client.create_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '38c53776-c450-4b86-a173-6e245863549a',
        'predictor_host': '127.0.0.1:30000',
        'train_job_id': '65af28c7-e3ef-4fb0-af76-8b413d16ad76'}