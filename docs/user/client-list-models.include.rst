Example:

    .. code-block:: python

        client.get_models_of_task(task='IMAGE_CLASSIFICATION')

    Output:

    .. code-block:: python

        [{'datetime_created': 'Sun, 18 Nov 2018 09:56:03 GMT',
        'dependencies': {'tensorflow': '1.12.0'},
        'docker_image': 'rafikiai/rafiki_worker:0.0.7',
        'model_class': 'TfFeedForward',
        'name': 'TfFeedForward',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': '9fdefa23-c838-4c56-8eb5-f625ff4245ab'},
        {'datetime_created': 'Sun, 18 Nov 2018 09:56:04 GMT',
        'dependencies': {'scikit-learn': '0.20.0'},
        'docker_image': 'rafikiai/rafiki_worker:0.0.7',
        'model_class': 'SkDt',
        'name': 'SkDt',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': '9fdefa23-c838-4c56-8eb5-f625ff4245ab'}]

.. seealso:: :meth:`rafiki.client.Client.get_models_of_task`
    

