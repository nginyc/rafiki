Quick Start (Admins)
====================================================================

As an *Admin*, you can manage users, datasets, models, train jobs & inference jobs on Rafiki. This guide only highlights the key methods available to manage users.

To learn about how to manage models, go to :ref:`quickstart-model-developers`.

To learn about how to manage train & inference jobs, go to :ref:`quickstart-app-developers`.

This guide assumes that you have access to a running instance of *Rafiki Admin* at ``<rafiki_host>:<admin_port>``
and *Rafiki Web Admin* at ``<rafiki_host>:<web_admin_port>``.

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
            email='admin@rafiki',
            password='rafiki',
            user_type='ADMIN'
        )
        
        client.create_user(
            email='model_developer@rafiki',
            password='rafiki',
            user_type='MODEL_DEVELOPER'
        )

        client.create_user(
            email='app_developer@rafiki',
            password='rafiki',
            user_type='APP_DEVELOPER'
        )


.. seealso:: :meth:`rafiki.client.Client.create_user`


Listing all users
--------------------------------------------------------------------

Example:

    .. code-block:: python

        client.get_users()
    

    .. code-block:: python

        [{'email': 'superadmin@rafiki',
        'id': 'c815fa08-ce06-467d-941b-afc27684d092',
        'user_type': 'SUPERADMIN'},
        {'email': 'admin@rafiki',
        'id': 'cb2c0d61-acd3-4b65-a5a7-d78aa5648283',
        'user_type': 'ADMIN'},
        {'email': 'model_developer@rafiki',
        'id': 'bfe58183-9c69-4fbd-a7b3-3fdc267b3290',
        'user_type': 'MODEL_DEVELOPER'},
        {'email': 'app_developer@rafiki',
        'id': '958a7d65-aa1d-437f-858e-8837bb3ecf32',
        'user_type': 'APP_DEVELOPER'}]
        

.. seealso:: :meth:`rafiki.client.Client.get_users`


Banning a user
--------------------------------------------------------------------

Example:

    .. code-block:: python

        client.ban_user('app_developer@rafiki')
    

.. seealso:: :meth:`rafiki.client.Client.ban_user`