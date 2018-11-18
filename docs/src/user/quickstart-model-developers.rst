.. _`quickstart-model-developers`:

Quick Start (Model Developers)
====================================================================

.. contents:: Table of Contents

We assume that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
and *Rafiki Admin Web* at ``<rafiki_host>:<admin_web_port>``.

Installing the Client
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the Client
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


Listing models by task
--------------------------------------------------------------------

.. include:: ./client-list-models.include.rst