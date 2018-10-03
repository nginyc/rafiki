.. _`quickstart-model-developers`:

Quickstart (Model Developers)
====================================================================

.. contents:: Table of Contents

This quickstart submits the example model of a fully-connected neural network with a single hidden layer, written in Tensorflow (``examples/models/SingleHiddenLayerTensorflowModel.py``).

Installation
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the Client
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.login`

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client()
        client.login(email='model_developer@rafiki', password='rafiki')

        
Creating models
--------------------------------------------------------------------

To create a model, you will need to write a model class that extends :class:`rafiki.model.BaseModel` in a single Python file.
You can validate your model definition with :meth:`rafiki.model.test_model_class`.

The base Rafiki worker image has the following Python libraries pre-installed:

::

    numpy==1.14.5 tensorflow==1.10.1 h5py==2.8.0 torch==0.4.1 Keras==2.2.2 scikit-learn==0.20.0

You can optionally build a custom Docker image for the model training & inference and pass the argument for `docker_image`. 
This Docker image has to extend `rafikiai/rafiki_worker`.

.. seealso:: :meth:`rafiki.client.Client.create_model`

Example:

    .. code-block:: python

        client.create_model(
            name='single_hidden_layer_tf',
            task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
            model_file_path='examples/models/SingleHiddenLayerTensorflowModel.py',
            model_class='SingleHiddenLayerTensorflowModel'
        )

Listing models by task
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_models_of_task`


Example:

    .. code-block:: python

        client.get_models_of_task(task='IMAGE_CLASSIFICATION_WITH_ARRAYS')

    Output:

    .. code-block:: python

        [{'datetime_created': 'Tue, 25 Sep 2018 09:28:44 GMT',
        'docker_image': 'rafikiai/rafiki_worker',
        'model_class': 'SingleHiddenLayerTensorflowModel',
        'name': 'single_hidden_layer_tf',
        'task': 'IMAGE_CLASSIFICATION_WITH_ARRAYS',
        'user_id': '695a23b0-a488-4110-aa50-6476e53a0e4d'}]
    

