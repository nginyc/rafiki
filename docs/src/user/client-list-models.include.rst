Example:

    .. code-block:: python

        client.get_models_of_task(task='IMAGE_CLASSIFICATION')

    Output:

    .. code-block:: python

       [{'access_right': 'PRIVATE',
        'datetime_created': 'Mon, 17 Dec 2018 07:06:03 GMT',
        'dependencies': {'tensorflow': '1.12.0'},
        'docker_image': 'rafikiai/rafiki_worker:0.0.9',
        'model_class': 'TfFeedForward',
        'name': 'TfFeedForward',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': 'fb5671f1-c673-40e7-b53a-9208eb1ccc50'},
        {'access_right': 'PRIVATE',
        'datetime_created': 'Mon, 17 Dec 2018 07:06:03 GMT',
        'dependencies': {'scikit-learn': '0.20.0'},
        'docker_image': 'rafikiai/rafiki_worker:0.0.9',
        'model_class': 'SkDt',
        'name': 'SkDt',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': 'fb5671f1-c673-40e7-b53a-9208eb1ccc50'}]

.. seealso:: :meth:`rafiki.client.Client.get_models_of_task`
    

