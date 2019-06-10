To create a model training job, you'll need to submit your dataset and a target task (see :ref:`tasks`), together with your app's name.
You'll need to prepare your dataset in a format specified by the target task, and upload it to a publicly accessible URL. 

After creating a train job, you can monitor it on Rafiki Web Admin (see :ref:`using-web-admin`).

Refer to the parameters of :meth:`rafiki.client.Client.create_train_job()` for configuring how your train job runs on Rafiki, such as enabling GPU usage.

Example:

    .. code-block:: python

        client.create_train_job(
            app='fashion_mnist_app',
            task='IMAGE_CLASSIFICATION',
            train_dataset_uri='https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true',
            test_dataset_uri='https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true',
            budget={ 'MODEL_TRIAL_COUNT': 2 }
        )

    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': 'ec4db479-b9b2-4289-8086-52794ffc71c8'}

.. note::

    The datasets in the above example have been pre-processed to conform to the task's dataset specification. 
    The code that does this pre-processing from the original Fashion MNIST dataset is available at `./examples/datasets/image_classification/load_mnist_format.py <https://github.com/nginyc/rafiki/tree/master/examples/datasets/image_classification/load_mnist_format.py>`_.
    
.. seealso:: :meth:`rafiki.client.Client.create_train_job`