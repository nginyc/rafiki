Example:

    .. code-block:: python

        client.get_inference_jobs_of_app(app='fashion_mnist_app')

    Output:

    .. code-block:: python

      {'app': 'fashion_mnist_app',
        'app_version': 1,
        'datetime_started': 'Mon, 17 Dec 2018 07:15:12 GMT',
        'datetime_stopped': None,
        'id': '0477d03c-d312-48c5-8612-f9b37b368949',
        'predictor_host': '127.0.0.1:30000',
        'status': 'RUNNING',
        'train_job_id': 'ec4db479-b9b2-4289-8086-52794ffc71c8'}

.. seealso:: :meth:`singaauto.client.Client.get_inference_jobs_of_app`