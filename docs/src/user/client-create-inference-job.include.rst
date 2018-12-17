To create an model deployment job, you need to submit the app name associated with a *stopped* train job.
The inference job would be created from the best trials from the train job.

Example:

    .. code-block:: python

        client.create_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '74b8f43a-c4f8-4ebc-a643-18a879dbbd1d',
        'predictor_host': '127.0.0.1:30000',
        'train_job_id': '3f3b3bdd-43ac-4354-99a5-d4d86006b68a'}

.. seealso:: :meth:`rafiki.client.Client.create_inference_job`