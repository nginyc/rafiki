.. seealso:: :meth:`rafiki.client.Client.create_model`

.. seealso:: :ref:`creating-models`

To create a model, you will need to write a model class that extends :class:`rafiki.model.BaseModel` in a single Python file,
where the model's implementation conforms to a specific task (see :ref:`tasks`).

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
