
You'll first need to convert your dataset into a format specified by one of the tasks (see :ref:`tasks`), 
and split them into two files: one for training & one for validation.
After doing so, you'll create 2 corresponding datasets on Rafiki by uploading them from your filesystem.

Example (pre-processing step):

  .. code-block:: shell

        # Run this in shell
        python examples/datasets/image_files/load_mnist_format.py


Example:

    .. code-block:: python

        client.create_dataset(
            name='fashion_mnist_train',
            task='IMAGE_CLASSIFICATION',
            dataset_path='data/fashion_mnist_train.zip'
        )

        client.create_dataset(
            name='fashion_mnist_val',
            task='IMAGE_CLASSIFICATION',
            dataset_path='data/fashion_mnist_val.zip'
        )

    Output:

    .. code-block:: python

        {'id': 'ecf87d2f-6893-4e4b-8ed9-1d9454af9763', 
        'name': 'fashion_mnist_train', 
        'size_bytes': 36702897, 
        'task': 'IMAGE_CLASSIFICATION'}

        {'id': '7e9a2f8a-c61d-4365-ae4a-601e90892b88', 
        'name': 'fashion_mnist_val', 
        'size_bytes': 6116386, 
        'task': 'IMAGE_CLASSIFICATION'}

.. seealso:: :meth:`rafiki.client.Client.create_dataset`

.. note::

    The code that preprocesses the original Fashion MNIST dataset is available at `./examples/datasets/image_files/load_mnist_format.py <https://github.com/nginyc/rafiki/tree/master/examples/datasets/image_files/load_mnist_format.py>`_.
    