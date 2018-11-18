Example:

    .. code-block:: python

        client.get_inference_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

       [{'app': 'fashion_mnist_app',
        'app_version': 1,
        'datetime_started': 'Sun, 18 Nov 2018 10:04:13 GMT',
        'datetime_stopped': None,
        'id': '74b8f43a-c4f8-4ebc-a643-18a879dbbd1d',
        'predictor_host': '127.0.0.1:30000',
        'status': 'RUNNING',
        'train_job_id': '3f3b3bdd-43ac-4354-99a5-d4d86006b68a'}]

.. seealso:: :meth:`rafiki.client.Client.get_inference_jobs_of_app`