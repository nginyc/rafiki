
.. _`model-development`:

Model Development Guide
====================================================================

Rafiki leverages on a dynamic pool of model templates contributed by *Model Developers*.

As a *Model Developer*, you'll define a *Python class* that conforms to Rafiki's base model specification, and
submit it to Rafiki with the :meth:`rafiki.client.Client.create_model` method.

Implementing the Base Model Interface
--------------------------------------------------------------------

As an overview, your model template needs to provide the following logic for deployment on Rafiki:

- Definition of the space of your model's hyperparameters (*knob configuration*)
- Initialization of the model with a concrete set of hyperparameters (*knobs*)
- Training of the model given a (train) dataset on the local file system
- Evaluation of the model given a (validation) dataset onthe local file system
- Dumping of the model's parameters for serialization, after training
- Loading of the model with trained parameters
- Making batch predictions with the model, after being trained

Full details of Rafiki's base model interface is documented at :class:`rafiki.model.BaseModel`.
Your model implementation has to follow a specific task's specification (see :ref:`tasks`).

To aid your implementation, you can refer to :ref:`sample-models`.

Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After implementing your model, you'll use :meth:`rafiki.model.dev.test_model_class` to test your model. 
Refer to its documentation for more details on how to use it, or refer to the sample models' usage of the method. 

Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``utils.logger`` in the ``rafiki.model`` module provides a set of methods to log messages & metrics while your model is training.
These messages & metrics would be displayed on Rafiki Web Admin for monitoring & debugging purposes.
Refer to :class:`rafiki.model.LoggerUtils` for more details.

.. seealso:: :ref:`using-web-admin` 

Dataset Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``utils.dataset`` in the ``rafiki.model`` module provides a simple set of in-built dataset loading methods. 
Refer to :class:`rafiki.model.DatasetUtils` for more details.


Defining Hyperparameter Search Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Refer to :ref:`model-tuning` for the specifics of how you can tune your models on Rafiki. 


.. _`sample-models`:

Sample Models
--------------------------------------------------------------------

To illustrate how to write models for Rafiki, we have written the following:

    - Sample pre-processing logic to convert common dataset formats to Rafiki's own dataset formats in `./examples/datasets/ <https://github.com/nginyc/rafiki/tree/master/examples/datasets/>`_ 
    - Sample models in `./examples/models/ <https://github.com/nginyc/rafiki/tree/master/examples/models/>`_


Example: Testing Models for ``IMAGE_CLASSIFICATION``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download & pre-process the original Fashion MNIST dataset to the dataset format specified by ``IMAGE_CLASSIFICATION``:

    .. code-block:: shell

        python examples/datasets/image_files/load_fashion_mnist.py

2. Install the Python dependencies for the sample models:

    .. code-block:: shell

        pip install scikit-learn==0.20.0
        pip install tensorflow==1.12.0

3. Test the sample models in ``./examples/models/image_classification``:

    .. code-block:: shell

        python examples/models/image_classification/SkDt.py
        python examples/models/image_classification/TfFeedForward.py


Example: Testing Models for ``POS_TAGGING``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download & pre-process the subsample of the Penn Treebank dataset to the dataset format specified by ``POS_TAGGING``:

    .. code-block:: shell

        python examples/datasets/corpus/load_sample_ptb.py

2. Install the Python dependencies for the sample models:

    .. code-block:: shell

        pip install torch==0.4.1

3. Test the sample models in ``./examples/models/pos_tagging``:

    .. code-block:: shell

        python examples/models/pos_tagging/BigramHmm.py
        python examples/models/pos_tagging/PyBiLstm.py


.. _`configuring-model-environment`:

Configuring the Model's Environment
--------------------------------------------------------------------

Your model will be run in Python 3.6 with the following Python libraries pre-installed:

    .. code-block:: shell

        requests==2.20.0
        numpy==1.14.5
        Pillow==5.3.0

Additionally, you'll specify a list of Python dependencies to be installed for your model, 
prior to model training and inference. This is configurable with the ``dependencies`` option 
during model creation. These dependencies will be lazily installed on top of the worker's Docker image before your model's code is executed.
If the model is to be run on GPU, Rafiki would map dependencies to their GPU-supported versions, if supported. 
For example, ``{ 'tensorflow': '1.12.0' }`` will be installed as ``{ 'tensorflow-gpu': '1.12.0' }``.
Rafiki could also parse specific dependency names to install certain non-PyPI packages. 
For example, ``{ 'singa': '1.1.1' }`` will be installed as ``singa-cpu=1.1.1`` or ``singa-gpu=1.1.1`` using ``conda``.

Refer to the list of officially supported dependencies below. For dependencies that are not listed,
they will be installed as PyPI packages of the specified name and version.

=====================       =====================
**Dependency**              **Installation Command**
---------------------       ---------------------        
``tensorflow``              ``pip install tensorflow==${ver}`` or ``pip install tensorflow-gpu==${ver}``
``singa``                   ``conda install -c nusdbsystem singa-cpu=${ver}`` or ``conda install -c nusdbsystem singa-gpu=${ver}``
``Keras``                   ``pip install Keras==${ver}``
``scikit-learn``            ``pip install scikit-learn==${ver}``
``torch``                   ``pip install torch==${ver}``
=====================       =====================

Alternatively, you can build a custom Docker image that extends ``rafikiai/rafiki_worker``,
installing the required dependencies for your model. This is configurable with ``docker_image`` option
during model creation.

.. seealso:: :meth:`rafiki.client.Client.create_model`

Your model should be GPU-sensitive based on the environment variable ``CUDA_AVAILABLE_DEVICES`` 
(see `here <https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/>`_).  
If ``CUDA_AVAILABLE_DEVICES`` is set to ``-1``, your model should simply run on CPU. 
You can assume that your model has exclusive access to the GPUs listed in ``CUDA_AVAILABLE_DEVICES``. 

