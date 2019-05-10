
.. _`creating-models`:

Creating Models
====================================================================

.. contents:: Table of Contents

To create a model, you will need to submit a model class that conforms to the specification
by :class:`rafiki.model.BaseModel`, written in a `single` Python file.
The model's implementation should conform to a specific task (see :ref:`tasks`).
To submit the model to Rafiki, use the :meth:`rafiki.client.Client.create_model` method.

Implementing Models
--------------------------------------------------------------------

Full details on how to implement a model are located in the documentation of :class:`rafiki.model.BaseModel`,
and sample model implementations are located in `./examples/models/ <https://github.com/nginyc/rafiki/tree/master/examples/models/>`_.

In defining the hyperparameters (knobs) of a model, refer to the documentation at :ref:`knob-types` for the full list of knob types.

After implementing your model, it is highly recommended to use :meth:`rafiki.model.test_model_class` 
to test your model. This method simulates a full train-inference flow on your model, ensuring that 
it is likely to work on Rafiki.

Logging in Models
--------------------------------------------------------------------

By importing the global ``logger`` instance in the ``rafiki.model`` module, 
you can log messages and metrics while your model is being trained, and you can 
define plots to visualize your model's training on Rafiki's Admin Web interface.

Refer to :class:`rafiki.model.ModelLogger` for full usage instructions.

.. seealso:: :ref:`using-admin-web` 

Dataset Loading in Models
--------------------------------------------------------------------

The global ``dataset_utils`` instance in the ``rafiki.model`` module provides
a set of built-in dataset loading methods for common dataset types on Rafiki.

Refer to :class:`rafiki.model.ModelDatasetUtils` for full usage instructions.

Model Environment
--------------------------------------------------------------------

Your model will be run in Python 3.6 with the following Python libraries pre-installed:

    .. code-block:: shell

        pip install numpy==1.14.5


Additionally, you'll specify a list of dependencies to be installed for your model, 
prior to model training and inference. This is configurable with the ``dependencies`` option 
during model creation. 

Alternatively, you can build a custom Docker image that extends ``rafikiai/rafiki_worker``,
installing the required dependencies for your model. This is configurable with ``docker_image`` option
during model creation.

Models should run at least run on CPU-only machines and optionally leverage on a shared GPU, if it is available.

Refer to the parameters of :meth:`rafiki.client.Client.create_model` for configuring how your model runs on Rafiki.

Sample Models
--------------------------------------------------------------------

To illustrate how to write models on Rafiki, we have written the following:

    - Sample pre-processing logic to convert common dataset formats to Rafiki's own dataset formats in `./examples/datasets/ <https://github.com/nginyc/rafiki/tree/master/examples/datasets/>`_ 
    - Sample models in `./examples/models/ <https://github.com/nginyc/rafiki/tree/master/examples/models/>`_

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
        python examples/models/image_classification/TfFeedForward.py


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
