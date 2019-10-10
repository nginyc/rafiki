Send a ``POST /predict`` to ``predictor_host`` with a body of the following format in JSON:

    ::

        {
            "query": <query>
        }

...where the format of ``<query>`` depends on the associated task (see :ref:`tasks`).

The body of the response will be of the following format in JSON:

    ::

        {
            "prediction": <prediction>
        }

...where the format of ``<prediction>`` depends on the associated task.


Example:

    If ``predictor_host`` is ``127.0.0.1:30000``, run the following in Python:

        .. code-block:: python
        
            predictor_host = '127.0.0.1:30000'
            query_path = 'examples/data/image_classification/fashion_mnist_test_1.png'

            # Load query image as 3D list of pixels
            from rafiki.model import utils
            [query] = utils.dataset.load_images([query_path]).tolist()

            # Make request to predictor
            import requests
            res = requests.post('http://{}/predict'.format(predictor_host), json={ 'query': query })
            print(res.json())

    Output:

        .. code-block:: python

            {'prediction': [0.9364003576825639, 1.016065009906697e-08, 0.0027604885399341583, 0.00014587241457775235, 6.018594376655528e-06, 1.042887332047826e-09, 0.060679372351310566, 2.024707311532037e-11, 7.901770004536957e-06, 1.5299328026685544e-08], 
            'predictions': []}
