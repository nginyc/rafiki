.. _`quickstart-model-developers`:

Quick Start (Model Developers)
====================================================================

As a *Model Developer*, you can manage models, datasets, train jobs & inference jobs on Rafiki. This guide only highlights the key methods available to manage models.

To learn about how to manage datasets, train jobs & inference jobs, go to :ref:`quickstart-app-developers`.

This guide assumes that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
and *Rafiki Web Admin* at ``<rafiki_host>:<web_admin_port>``.

To learn more about what else you can do on Rafiki, explore the methods of :class:`rafiki.client.Client`

Installing the client
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the client
--------------------------------------------------------------------

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client(admin_host='localhost', admin_port=3000)
        client.login(email='model_developer@rafiki', password='rafiki')

.. seealso:: :meth:`rafiki.client.Client.login`        

Creating models
--------------------------------------------------------------------

.. include:: ./client-create-models.include.rst


Listing available models by task
--------------------------------------------------------------------

.. include:: ./client-list-models.include.rst


Deleting a model
--------------------------------------------------------------------

Example:

    .. code-block:: python

        client.delete_model('fb5671f1-c673-40e7-b53a-9208eb1ccc50')

.. seealso:: :meth:`rafiki.client.Client.delete_model`        
