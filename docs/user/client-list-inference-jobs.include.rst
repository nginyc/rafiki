.. seealso:: :meth:`rafiki.client.Client.get_inference_jobs_of_app`

Example:

    .. code-block:: python

        client.get_inference_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        [{'app': 'fashion_mnist_app',
        'app_version': 1,
        'datetime_started': 'Thu, 04 Oct 2018 03:31:59 GMT',
        'datetime_stopped': None,
        'id': '38c53776-c450-4b86-a173-6e245863549a',
        'predictor_host': '127.0.0.1:30000',
        'status': 'RUNNING',
        'train_job_id': '65af28c7-e3ef-4fb0-af76-8b413d16ad76'}]