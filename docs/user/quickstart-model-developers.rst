.. _`quickstart-model-developers`:

Quickstart (Model Developers)
====================================================================

.. contents:: Table of Contents

Installation
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the Client
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.login`

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client(admin_host='localhost', admin_port=8000)
        client.login(email='superadmin@rafiki', password='rafiki')

        
Creating models
--------------------------------------------------------------------

To create a model, you will need to write a model class that extends :class:`rafiki.model.BaseModel` in a single Python file.
Refer to sample model definitions in `./examples/models/` folder of the project, and validate your model definition with :meth:`rafiki.model.validate_model_class`.

The base Rafiki worker image runs Python 3.6 and the following Python libraries pre-installed:

::

    pip install numpy==1.14.5 tensorflow==1.10.1 h5py==2.8.0 torch==0.4.1 Keras==2.2.2 scikit-learn==0.20.0

.. seealso:: :meth:`rafiki.client.Client.create_model`

Examples:

    .. code-block:: python

        client.create_model(
            name='TfSingleHiddenLayer',
            task='IMAGE_CLASSIFICATION',
            model_file_path='examples/models/image_classification/TfSingleHiddenLayer.py',
            model_class='TfSingleHiddenLayer'
        )

        client.create_model(
            name='SkDt',
            task='IMAGE_CLASSIFICATION',
            model_file_path='examples/models/image_classification/SkDt.py',
            model_class='SkDt'
        )

.. note::

    If you have additional Python dependencies, you can build a custom Docker image for model training & inference and pass an additional argument `docker_image`.
    This Docker image should extend `rafikiai/rafiki_worker`.


Listing models by task
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.get_models_of_task`


Example:

    .. code-block:: python

        client.get_models_of_task(task='IMAGE_CLASSIFICATION')

    Output:

    .. code-block:: python

        [{'datetime_created': 'Thu, 04 Oct 2018 03:24:58 GMT',
        'docker_image': 'rafikiai/rafiki_worker:0.0.3',
        'model_class': 'TfSingleHiddenLayer',
        'name': 'TfSingleHiddenLayer',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': '23f3526a-35d1-46ba-be68-af8f4992a0f9'},
        {'datetime_created': 'Thu, 04 Oct 2018 03:24:59 GMT',
        'docker_image': 'rafikiai/rafiki_worker:0.0.3',
        'model_class': 'SkDt',
        'name': 'SkDt',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': '23f3526a-35d1-46ba-be68-af8f4992a0f9'}]
    

