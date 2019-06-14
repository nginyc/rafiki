.. _`quick-start`:

Quick Start
====================================================================

This guide assumes you have deployed your own empty instance of Rafiki and you want to try a *full* train-inference flow as the *Super Admin*, 
including adding of models, submitting model training & serving jobs, and making predictions on Rafiki.


.. note::

    - For *Model Developers* just looking to contribute models, refer to :ref:`quickstart-model-developers`
    - For *Application Developers* just looking to train and deploy models, refer to :ref:`quickstart-app-developers`
    - For *Application Users* just looking to make predictions, refer to :ref:`quickstart-app-users`

.. note::

    Refer to :ref:`user-types` to understand more about the different types of users on Rafiki.

The sequence of examples below submits the `Fashion MNIST dataset <https://github.com/zalandoresearch/fashion-mnist>`_ for training and inference. 
Alternatively, after installing the Rafiki Client's dependencies, you can refer and run the scripted version of this quickstart 
`./examples/scripts/quickstart.py <https://github.com/nginyc/rafiki/blob/master/examples/scripts/quickstart.py>`_.

.. note::

    If you haven't set up Rafiki on your local machine, refer to :ref:`quick-setup` before continuing.


To learn more about what you can do on Rafiki, explore the methods of :class:`rafiki.client.Client`.

Installing the client
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the client
--------------------------------------------------------------------

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client(admin_host='localhost', admin_port=3000)
        client.login(email='superadmin@rafiki', password='rafiki')


.. seealso:: :meth:`rafiki.client.Client.login`

Creating models
--------------------------------------------------------------------

.. include:: ./client-create-models.include.rst


Listing available models by task
--------------------------------------------------------------------

.. include:: ./client-list-models.include.rst


Creating a train job
--------------------------------------------------------------------

.. include:: ./client-create-train-job.include.rst


Listing train jobs
--------------------------------------------------------------------

.. include:: ./client-list-train-jobs.include.rst


Creating an inference job with the latest train job
--------------------------------------------------------------------

.. include:: ./client-create-inference-job.include.rst


Listing inference jobs
--------------------------------------------------------------------

.. include:: ./client-list-inference-jobs.include.rst


Making predictions
--------------------------------------------------------------------

.. include:: ./making-predictions.include.rst


Stopping a running inference job
--------------------------------------------------------------------

.. include:: ./client-stop-inference-job.include.rst