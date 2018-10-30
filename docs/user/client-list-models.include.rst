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
    

