.. seealso:: :meth:`rafiki.client.Client.create_model`

To create a model, you will need to write a model class that extends :class:`rafiki.model.BaseModel` in a single Python file.
Refer to sample model definitions in `./examples/models <https://github.com/nginyc/rafiki/blob/master/examples/models>`_ folder of the project, 
and validate your model definition with :meth:`rafiki.model.validate_model_class`.

The base Rafiki worker image runs Python 3.6 and the following Python libraries pre-installed:

::

    pip install numpy==1.14.5 tensorflow==1.10.1 h5py==2.8.0 torch==0.4.1 Keras==2.2.2 scikit-learn==0.20.0


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
