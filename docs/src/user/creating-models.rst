
.. _`creating-models`:

Creating Models
====================================================================

.. contents:: Table of Contents


To create a model on Rafiki, use the :meth:`rafiki.client.Client.create_model` method.


Model Environment
--------------------------------------------------------------------

Your model will be run in Python 3.6 with the following Python libraries pre-installed:

    .. code-block:: shell

        pip install numpy==1.14.5


Additionally, you'll specify a list of dependencies to be installed for your model, 
prior to model training and inference. This is configurable with the ``dependencies`` option 
during model creation. 

Alternatively, you can build a custom Docker image that extends ``rafikiai/rafiki_worker``,
installing the required dependencies for your model. This is configurable with ``docker_image``) option
during model creation.

Models should run at least run on CPU-only machines and optionally leverage on a shared GPU, if it is available.

Refer to the parameters of :meth:`rafiki.client.Client.create_model` for configuring how your model runs on Rafiki.

Testing Models
--------------------------------------------------------------------

To illustrate how to write models on Rafiki, we have written the following:

    - Sample pre-processing logic to convert common dataset formats to Rafiki's own dataset formats in `./examples/datasets/ <https://github.com/nginyc/rafiki/tree/master/examples/datasets/>`_ 
    - Sample models in `./examples/models/ <https://github.com/nginyc/rafiki/tree/master/examples/models/>`_
    - A method :meth:`rafiki.model.test_model_class` that simulates a full train-inference flow on any Rafiki model 

To start testing your model, first install the Python dependencies at ``rafiki/model/requirements.txt``:

    .. code-block:: shell

        pip install -r rafiki/model/requirements.txt


Example: Testing Models for ``IMAGE_CLASSIFICATION``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download & pre-process the original Fashion MNIST dataset to the dataset format specified by ``IMAGE_CLASSIFICATION``:

    .. code-block:: shell

        python examples/datasets/image_classification/load_mnist_format.py

2. Install the Python dependencies for the sample models:

    .. code-block:: shell

        pip install scikit-learn==0.20.0
        pip install tensorflow==1.12.0

3. Test the sample models in ``./examples/models/image_classification`` with :meth:`rafiki.model.test_model_class`:

    .. code-block:: shell

        python examples/models/image_classification/SkDt.py
        python examples/models/image_classification/TfSingleHiddenLayer.py


Example: Testing Models for ``POS_TAGGING``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download & pre-process the subsample of the Penn Treebank dataset to the dataset format specified by ``POS_TAGGING``:

    .. code-block:: shell

        python examples/datasets/pos_tagging/load_ptb_format.py

2. Install the Python dependencies for the sample models:

    .. code-block:: shell

        pip install torch==0.4.1

3. Test the sample models in ``./examples/models/pos_tagging`` with :meth:`rafiki.model.test_model_class`:

    .. code-block:: shell

        python examples/models/pos_tagging/BigramHmm.py
        python examples/models/pos_tagging/PyBiLstm.py


Model Logging & Dataset Loading
--------------------------------------------------------------------

:class:`rafiki.model.BaseModel` has a property ``utils`` that subclasses the model utility classes
:class:`rafiki.model.log.ModelLogUtils` and :class:`rafiki.model.dataset.ModelDatasetUtils`. They 
help with model logging & dataset loading respectively. 

Refer to the sample usage in the implementation of `./examples/models/image_classification/TfSingleHiddenLayer.py <https://github.com/nginyc/rafiki/tree/master/examples/models/image_classification/TfSingleHiddenLayer.py>`_.