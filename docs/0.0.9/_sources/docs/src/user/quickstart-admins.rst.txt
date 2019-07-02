Quick Start (Admins)
====================================================================

.. contents:: Table of Contents

As an Admin, you can manage users, manage models, and manage train & inference jobs on Rafiki.

This quickstart only highlights the key methods available to manage users.
To learn about how to manage models, go to :ref:`quickstart-model-developers`,
To learn about how to manage train & inference jobs, go to :ref:`quickstart-app-developers`,
To learn more about what you can do on Rafiki, explore the methods of :class:`rafiki.client.Client`.

We assume that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
and *Rafiki Admin Web* at ``<rafiki_host>:<admin_web_port>``.

Installation
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
        
Creating users
--------------------------------------------------------------------

Examples:

    .. code-block:: python

        client.create_user(
            email='app_developer@rafiki',
            password='rafiki',
            user_type='APP_DEVELOPER'
        )

        client.create_user(
            email='model_developer@rafiki',
            password='rafiki',
            user_type='MODEL_DEVELOPER'
        )


.. seealso:: :meth:`rafiki.client.Client.create_user`