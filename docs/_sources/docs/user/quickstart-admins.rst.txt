Quickstart (Rafiki Admins)
====================================================================

.. contents:: Table of Contents


Installation
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the Client
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.login`

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client(admin_host='localhost', admin_port=8000)
        client.login(email='superadmin@rafiki', password='rafiki')
        
Creating users
--------------------------------------------------------------------

.. seealso:: :meth:`rafiki.client.Client.create_user`

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


