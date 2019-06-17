Example:

    .. code-block:: python

        client.get_available_models(task='IMAGE_CLASSIFICATION')

    Output:

    .. code-block:: python

       [{'access_right': 'PRIVATE',
        'datetime_created': 'Mon, 17 Dec 2018 07:06:03 GMT',
        'dependencies': {'tensorflow': '1.12.0'},
        'id': '45df3f34-53d7-4fb8-a7c2-55391ea10030',
        'name': 'TfFeedForward',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': 'fb5671f1-c673-40e7-b53a-9208eb1ccc50'},
        {'access_right': 'PRIVATE',
        'datetime_created': 'Mon, 17 Dec 2018 07:06:03 GMT',
        'dependencies': {'scikit-learn': '0.20.0'},
        'id': 'd0ea96ce-478b-4167-8a84-eb36ae631235',
        'name': 'SkDt',
        'task': 'IMAGE_CLASSIFICATION',
        'user_id': 'fb5671f1-c673-40e7-b53a-9208eb1ccc50'}]

.. seealso:: :meth:`rafiki.client.Client.get_available_models`
