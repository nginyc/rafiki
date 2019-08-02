
To create a model, you'll need to submit a model class that conforms to the specification
by :class:`rafiki.model.BaseModel`, written in a `single` Python file.
The model's implementation should conform to a specific task (see :ref:`tasks`).

Refer to the parameters of :meth:`rafiki.client.Client.create_model` for configuring how your model runs on Rafiki,
and refer to :ref:`model-development` to understand more about how to write & test models for Rafiki.

Example:

    .. code-block:: python

        client.create_model(
            name='TfFeedForward',
            task='IMAGE_CLASSIFICATION',
            model_file_path='examples/models/image_classification/TfFeedForward.py',
            model_class='TfFeedForward',
            dependencies={ 'tensorflow': '1.12.0' }
        )

        client.create_model(
            name='SkDt',
            task='IMAGE_CLASSIFICATION',
            model_file_path='examples/models/image_classification/SkDt.py',
            model_class='SkDt',
            dependencies={ 'scikit-learn': '0.20.0' }
        )

.. seealso:: :meth:`rafiki.client.Client.create_model`
