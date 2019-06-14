To create a model training job, you'll specify the train & validation datasets by their IDs, together with your application's name and its associated task.

After creating a train job, you can monitor it on Rafiki Web Admin (see :ref:`using-web-admin`).

Refer to the parameters of :meth:`rafiki.client.Client.create_train_job()` for configuring how your train job runs on Rafiki, such as enabling GPU usage & specifying which models to use.

Example:

    .. code-block:: python

        client.create_train_job(
            app='fashion_mnist_app',
            task='IMAGE_CLASSIFICATION',
            train_dataset_id='ecf87d2f-6893-4e4b-8ed9-1d9454af9763',
            val_dataset_id='7e9a2f8a-c61d-4365-ae4a-601e90892b88',
            budget={ 'MODEL_TRIAL_COUNT': 5 }
        )

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': 'ec4db479-b9b2-4289-8086-52794ffc71c8'}

.. seealso:: :meth:`rafiki.client.Client.create_train_job`