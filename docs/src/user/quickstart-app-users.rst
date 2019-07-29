.. _`quickstart-app-users`:

Quick Start (Application Users)
====================================================================

As an App User, you can make predictions on models deployed on Rafiki.

.. _`making-predictions`:


Making a single prediction
--------------------------------------------------------------------

.. seealso:: :ref:`creating-inference-job`

Your app developer should have created an inference job and shared *predictor_host*, the host at which you can send queries
to and receive predictions over HTTP. 

.. include:: ./making-predictions.include.rst


Making batch predictions
--------------------------------------------------------------------

Similar to making a single prediction, but use the ``queries`` attribute instead of ``query`` in your request and 
pass an *array* of queries instead.

Example:

    If ``predictor_host`` is ``127.0.0.1:30000``, run the following in Python:

        .. code-block:: python
        
            predictor_host = '127.0.0.1:30000'
            query_paths = ['examples/data/image_classification/fashion_mnist_test_1.png', 
                        'examples/data/image_classification/fashion_mnist_test_2.png']

            # Load query image as 3D list of pixels
            from rafiki.model import utils
            queries = utils.dataset.load_images(query_paths).tolist()

            # Make request to predictor
            import requests
            res = requests.post('http://{}/predict'.format(predictor_host), json={ 'queries': queries })
            print(res.json())

    Output:

        .. code-block:: python

            {'prediction': None, 
            'predictions': [[0.9364002384732744, 1.0160608354681244e-08, 0.0027604878414422274, 0.0001458720798837021, 6.018587100697914e-06, 1.0428869989809186e-09, 0.06067946175827773, 2.0247028012509993e-11, 7.901745448180009e-06, 1.5299294275905595e-08], [0.866741563402005, 5.757699909736402e-05, 0.0006144539802335203, 0.03480150588134776, 3.4249271266162395e-05, 1.3578344004727683e-09, 0.09774905198545598, 6.071191726436664e-12, 1.5324986861742218e-06, 1.583319586551113e-10]]}
