Quick Start (Admins)
====================================================================

As an *Admin*, you can manage users, datasets, models, train jobs & inference jobs on Rafiki. This guide only highlights the key methods available to manage users.

To learn about how to manage models, go to :ref:`quickstart-model-developers`.

To learn about how to manage train & inference jobs, go to :ref:`quickstart-app-developers`.

This guide assumes that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
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