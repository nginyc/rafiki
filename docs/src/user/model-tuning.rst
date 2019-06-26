
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
You'll define a search space of hyperparameters (*knob config*) in a declarative manner with the static method :meth:`rafiki.model.BaseModel.get_knob_config`.
The method should return a mapping of hyperparameter names (*knob names*) to hyperparameter specifications (*knob specifications*). 
A hyperparameter specification is an instance of a class that extends :class:`rafiki.model.BaseKnob`, which is limited to any of the following:

- :class:`rafiki.model.FixedKnob`
- :class:`rafiki.model.CategoricalKnob` 
- :class:`rafiki.model.FloatKnob`
- :class:`rafiki.model.IntegerKnob` 
- :class:`rafiki.model.PolicyKnob`
- :class:`rafiki.model.ArchKnob`

Refer to their documentation for more details on each type of knob specification.

.. _`model-policies`:

Model Policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`rafiki.model.PolicyKnob` is a special type of knob specification that allows Rafiki to configure the *behaviour* of a model on a *trial basis*.

In a modern model hyperparameter tuning scheme, a model tends to switch between different "modes", or so we call *policies*. For example,
when you tune your model manually, you might want the model to do early-stopping for the first e.g. 100 trials, then conduct a final trial for a full e.g. 300 epochs.
As such, the concept of model policies in Rafiki enables Rafiki's tuning advisor to externally configure your model to switch between these "modes".

Your model communicates to Rafiki which policies it supports by adding ``PolicyKnob(policy_name)`` to your model's knob_configuration. 
On the other hand, during training, Rafiki configures the *activation* of the model's policies on a trial basis 
by *realising* the values of ``PolicyKnob``s to either ``True`` (activated) or ``False`` (not activated).

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

Refer to the next section of :ref:`model-tuning-schemes` to better understand which policies your model should support to optimize hyperparameter search for your model.

.. _`model-tuning-schemes`:

Model Tuning Schemes
====================================================================

At a model level, Rafiki *automatically* selects the appropriate tuning scheme (*advisor*) based on the composition of the model's knob configuration.
Specifically, it employs the following rules, in the *given order*, to select the type of advisor to use:

+-----------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| **Rule**                                      | Tuning Scheme                                                                                             |
+===============================================+===================================================================+=======================================+
| | Only ``FixedKnob``                          | Only conduct a single trial                                                                               |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| | Only ``PolicyKnob``, ``FixedKnob``,         | | Hyperparameter tuning with Bayesian Optimization & cross-trial parameter sharing.                       |
| | ``FloatKnob``, ``IntegerKnob``,             | | Share globally best-scoring parameters across workers in a epsilon greedy manner.                       |
| | ``CategoricalKnob``, with policy            | | Optionally employ early stopping (``EARLY_STOP`` policy) for all trials.                                |  
| | ``SHARE_PARAMS``                            |                                                                                                           |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| | Only ``PolicyKnob``, ``FixedKnob``,         | | Hyperparameter tuning with Bayesian Optimization. Optionally employ early stopping                      | 
| | ``FloatKnob``, ``IntegerKnob``,             | | (``EARLY_STOP`` policy) for all trials, except for the final trial.                                     |
| | ``CategoricalKnob``                         |                                                                                                           |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| | Only ``PolicyKnob``, ``FixedKnob``,         | | Architecture tuning with cell-based                                                                     |
| | ``ArchKnob``, with policies                 | | `"Efficient Neural Architecture Search via Parameter Sharing" <https://arxiv.org/abs/1802.03268>`_.     |
| | ``SHARE_PARAMS``, ``EARLY_STOP``            |                                                                                                           |
| | ``SKIP_TRAIN``, ``QUICK_EVAL``              |                                                                                                           |
| | ``DOWNSCALE``                               |                                                                                                           |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
| All others                                    | Hyperparameter tuning with uniformly random knobs                                                         |
+-----------------------------------------------+-------------------------------------------------------------------+---------------------------------------+
