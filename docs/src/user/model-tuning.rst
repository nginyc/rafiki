
.. _`model-tuning`:

How Model Tuning Works
--------------------------------------------------------------------
Traditionally, getting the best performing model on a dataset involves involves *tedious* manual hyperparameter tuning.
On Rafiki, model hyperparameter tuning is automated by conducting multiple *trials* in a train job. 

Over the trials, the model is initialized with different hyperparameters (*knobs*), trained and evaluated.
A hyperparameter tuning *advisor* on Rafiki ingests the *validation scores* from these trials to suggest better hyperparameters for future trials,
to maximise performance of a model on the dataset.
At the very end of the train job, Rafiki could deploy the best-scoring trials for predictions.

Defining Hyperparameter Search Space
====================================================================
You'll define a search space of hyperparameters (*knob configuration*) in a declarative manner with the static method :meth:`rafiki.model.BaseModel.get_knob_config`.
The method should return a mapping of hyperparameter names (*knob names*) to hyperparameter specifications (*knob specifications*). 
A hyperparameter specification is an instance of a class that extends :class:`rafiki.model.BaseKnob`, which is limited to any of the following:

- :class:`rafiki.model.FixedKnob`
- :class:`rafiki.model.CategoricalKnob` 
- :class:`rafiki.model.FloatKnob`
- :class:`rafiki.model.IntegerKnob` 
- :class:`rafiki.model.PolicyKnob`
- :class:`rafiki.model.ArchKnob`

Refer to their documentation for more details on each type of knob specification, and refer to :ref:`sample-models` to see examples of 
how knob configurations are declared.

.. _`model-policies`:

Model Policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`rafiki.model.PolicyKnob` is a special type of knob specification that allows Rafiki to configure the *behaviour* of a model on a *trial basis*.

In a modern model hyperparameter tuning scheme, a model tends to switch between different "modes", or so we call *policies*. For example,
when you tune your model manually, you might want the model to do early-stopping for the first e.g. 100 trials, then conduct a final trial for a full e.g. 300 epochs.
As such, the concept of model policies in Rafiki enables Rafiki's tuning advisor to externally configure your model to switch between these "modes".

Your model communicates to Rafiki which policies it supports by adding ``PolicyKnob(policy_name)`` to your model's knob_configuration. 
On the other hand, during training, Rafiki configures the *activation* of the model's policies on a trial basis 
by *realising* the values of ``PolicyKnob`` to either ``True`` (activated) or ``False`` (not activated).

For example, if Rafiki's tuning scheme for your model requires your model to engage in e.g. early-stopping for all trials except for the final trial, 
if your model has ``{ 'early_stop': PolicyKnob('EARLY_STOP'), ... }``, Rafiki will pass ``early_stop=False`` for just the final trial as part of its knobs, and 
pass ``early_stop=True`` for all other trials. Your model would *situationally* do early-stopping based on the value of the knob `early-stop`.

Below is the list of officially recognized model policies:

+------------------------------+--------------------------------------------------------------------------------------------------------------------+
| **Policy**                   | Description                                                                                                        |
+==============================+====================================================================================================================+
| ``SHARE_PARAMS``             | Whether model should load the shared parameters passed in ``train()``                                              | 
+------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``EARLY_STOP``               | Whether model should stop training early in ``train()``, e.g. with use of early stopping or reduced no. of epochs  |
+------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``SKIP_TRAIN``               | Whether model should skip training its parameters                                                                  |
+------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``QUICK_EVAL``               | Whether model should stop evaluation early in ``evaluate()``, e.g. by evaluating on only a subset of their         |
|                              | validation dataset                                                                                                 |
+------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``DOWNSCALE``                | Whether a smaller version of the model should be constructed e.g. with fewer layers                                |
+------------------------------+--------------------------------------------------------------------------------------------------------------------+


.. _`model-tuning-schemes`:

Model Tuning Schemes
====================================================================

At a model level, Rafiki *automatically* selects the appropriate tuning scheme (*advisor*) based on the composition of the model's knob configuration 
and the *incoming train job's budget*. 

Specifically, it employs the following rules, in the *given order*, to select the type of advisor to use:

+-----------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **Rule**                                      | Tuning Scheme                                                                                             |
+===============================================+===================================================================+=======================================+
| | Only ``PolicyKnob``, ``FixedKnob``          | Only conduct a single trial                                                                               |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| | Only ``PolicyKnob``, ``FixedKnob``,         | | Hyperparameter tuning with Bayesian Optimization & cross-trial parameter sharing.                       |
| | ``FloatKnob``, ``IntegerKnob``,             | | Share globally best-scoring parameters across workers in a epsilon greedy manner.                       |
| | ``CategoricalKnob``, with policy            | | Optionally employ early stopping (``EARLY_STOP`` policy) for all trials.                                |  
| | ``SHARE_PARAMS``                            | |                                                                                                         |
| |                                             | | More details at :ref:`tuning-with-param-sharing`.                                                       |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| | Only ``PolicyKnob``, ``FixedKnob``,         | | Hyperparameter tuning with Bayesian Optimization. Optionally employ early stopping                      | 
| | ``FloatKnob``, ``IntegerKnob``,             | | (``EARLY_STOP`` policy) before the last 1h, and perform standard trials during the last 1h.             |
| | ``CategoricalKnob``                         |                                                                                                           |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| | Only ``PolicyKnob``, ``FixedKnob``,         | | Architecture tuning with cell-based                                                                     |
| | ``ArchKnob``, with policies                 | | `"Efficient Neural Architecture Search via Parameter Sharing" <https://arxiv.org/abs/1802.03268>`_.     |
| | ``SHARE_PARAMS``, ``EARLY_STOP``            | | It conducts *ENAS architecture search* before the last 12h, then performs the final                     | 
| | ``SKIP_TRAIN``, ``QUICK_EVAL``              | | training of the best architectures found in the last 12h.                                               |
| | ``DOWNSCALE``, and ``TIME_HOURS`` budget    | |                                                                                                         |
| | >= 12h                                      | | More details at :ref:`arch-tuning-with-enas`.                                                           |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| All others                                    | Hyperparameter tuning with uniformly random knobs                                                         |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+

The following subsections briefly explain how to leverage on the various model tuning schemes on Rafiki.

Hyperparameter Tuning with Bayesian Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To tune the hyperparameters of your model, where the hyperparameters are *simply floats, integers or categorical*, use :class:`rafiki.model.FixedKnob`,
:class:`rafiki.model.CategoricalKnob`, :class:`rafiki.model.FloatKnob` & :class:`rafiki.model.IntegerKnob`. 


Hyperparameter Tuning with Bayesian Optimization & Early Stopping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To additionally employ early stopping during hyperparameter tuning to speed up the tuning process, declare an extra :class:`rafiki.model.PolicyKnob` of 
the ``EARLY_STOP`` policy (see :ref:`model-policies`). 

Refer to the sample model `./examples/models/image_classification/TfFeedForward.py <https://github.com/nginyc/rafiki/tree/master/examples/models/image_classification/TfFeedForward.py>`_.

.. _`tuning-with-param-sharing`:

Hyperparameter Tuning with Bayesian Optimization & Parameter Sharing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To additionally have *best-scoring* model parameters shared between trials to speed up the tuning process 
(as outlined in `"Rafiki: Machine Learning as an Analytics Service System" <https://arxiv.org/pdf/1804.06087.pdf>`_),
declare an extra :class:`rafiki.model.PolicyKnob` of the ``SHARE_PARAMS`` policy (see :ref:`model-policies`). 

Refer to the sample model `./examples/models/image_classification/PyDenseNetBc.py <https://github.com/nginyc/rafiki/tree/master/examples/models/image_classification/PyDenseNetBc.py>`_
and its corresponding usage script `./examples/scripts/image_classification/train_densenet.py  <https://github.com/nginyc/rafiki/tree/master/examples/scripts/image_classification/train_densenet.py>`_
to better understand how to do parameter sharing.

.. _`arch-tuning-with-enas`:

Architecture Tuning with ENAS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To tune the architecture for your model with the modern architecture search algorithm 
`"Efficient Neural Architecture Search via Parameter Sharing" <https://arxiv.org/abs/1802.03268>`_ (*ENAS*), 
declare a :class:`rafiki.model.ArchKnob` and offer the policies ``SHARE_PARAMS``, ``EARLY_STOP``, ``SKIP_TRAIN``, ``QUICK_EVAL`` and ``DOWNSCALE`` (see :ref:`model-policies`).
Specifically, you'll need your model to support parameter sharing, stopping training early, skipping the training step, evaluating
on a subset of the validation dataset, and *downscaling* the model e.g. to use fewer layers. These policies are critical in
the speed & performance of ENAS. See :ref:`enas` to understand more about Rafiki's implementation of ENAS.

Refer to the sample model `./examples/models/image_classification/TfEnas.py <https://github.com/nginyc/rafiki/tree/master/examples/models/image_classification/TfEnas.py>`_
and its corresponding usage script `./examples/scripts/image_classification/run_enas.py <https://github.com/nginyc/rafiki/tree/master/examples/scripts/image_classification/run_enas.py>`_
to better understand how to do architecture tuning.


.. _`enas`:

Deep Dive on ENAS 
====================================================================

The ENAS paper outlines a new methodology for automatic neural network construction, 
speeding up the original Neural Architecture Search (NAS) methodology by 1000x without affecting its ability to search for a competitive architecture. 
The authors made the crucial observation that 2 different architectures would share a common subgraph, 
and the model parameters in that subgraph could be reused across trials without having to re-train these parameters from scratch every trial. 

The following is an overview of how ENAS works.
As explained in the ENAS paper, during an ENAS search for best CNN architecture (*ENAS Search*), 
there is an alternation between 2 phases: training of the ENAS CNN’s shared parameters (*CNN Train Phase*), 
and the training of the ENAS controller (*Controller Train Phase*). While CNN parameters are carried over the phases, 
the CNN’s shared parameters are not trained during Controller Train Phases. 
After ENAS Search is done, there is a final training of the best CNN architecture found (*ENAS Train*), 
this time initializing its CNN parameters from scratch,

On Rafiki, we've replicated the *Cell-Based ENAS* controller for image classification as one of Rafiki's tuning scheme and
a Rafiki model ``TfEnas``, with very close reference to author’s code. In this specific setup for ENAS, 
ENAS Search is done with the construction of a single *supergraph* of all possible architectures, 
while ENAS Train is done with the construction of a *fixed graph* of the best architecture (with slight architectural differences from ENAS Search). 
Each CNN Train Phase involves training the CNN for 1 epoch, while within each Controller Train Phase, the controller is trained for 30 steps. 
In each controller step, 10 architectures are sampled from the controller, evaluated on the ENAS CNN by *dynamically changing its architecture*, 
and losses based on validation accuracies are back-propagated in the controller to update the controller’s parameters. 
Each validation accuracy is computed on only a *batch* of the validation dataset. 
The alternation between CNN Train Phase and Controller Train Phase happens for ``X`` cycles during ENAS Search, and close to 
the end of training, during ENAS Train, architecture samples with highest validation accuracies, this time computed on the *full* validation dataset, 
would be trained from scratch to arrive at final best models.

We've generalized the ENAS controller, its architecture encoding scheme and its overall tuning scheme on Rafiki, such that Rafiki models can 
leverage on architecture tuning with a flexible architecture encoding, and Rafiki's application developers can train with these models
in an end-to-end manner. 

We've also devised a simple, yet effective strategy to run ENAS in a *distributed* setting. When given multiple GPUs, Rafiki performs 
ENAS *locally at each worker* in a train job, with these workers sharing a central ENAS controller. 
