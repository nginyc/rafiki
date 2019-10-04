To create an model serving job, you'll have to wait for your train job to stop. 
Then, you'll submit the app name associated with the train job (with a status of ``STOPPED``).
The inference job would be created from the best trials from that train job.

Example:

    .. code-block:: python

        client.create_inference_job(app='fashion_mnist_app')

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': '0477d03c-d312-48c5-8612-f9b37b368949',
        'predictor_host': '127.0.0.1:30001',
        'train_job_id': 'ec4db479-b9b2-4289-8086-52794ffc71c8'}

.. seealso:: :meth:`singaauto.client.Client.create_inference_job`