.. _`development`:

Development
====================================================================

Before running any individual scripts, make sure to run the shell configuration script:

    .. code-block:: shell

        source .env.sh

Refer to :ref:`architecture` and :ref:`folder-structure` for a developer's overview of Rafiki.

Testing Latest Code Changes
--------------------------------------------------------------------

To test the lastet code changes e.g. in the ``dev`` branch, you'll need to do the following:

    1. Build Rafiki's images on each participating node (the quickstart instructions pull pre-built `Rafiki's images <https://hub.docker.com/r/rafikiai/>`_ from Docker Hub):

    .. code-block:: shell

        bash scripts/build_images.sh

    2. Purge all of Rafiki's data (since there might be database schema changes):

    .. code-block:: shell

        bash scripts/clean.sh


Managing Rafiki's DB
--------------------------------------------------------------------

By default, you can connect to the PostgreSQL DB using a PostgreSQL client (e.g `Postico <https://eggerapps.at/postico/>`_) with these credentials:

    ::

        RAFIKI_ADDR=127.0.0.1
        POSTGRES_EXT_PORT=5433
        POSTGRES_USER=rafiki
        POSTGRES_DB=rafiki
        POSTGRES_PASSWORD=rafiki


You can start & stop Rafiki's DB independently of the rest of Rafiki's stack with:

    .. code-block:: shell

        bash scripts/start_db.sh
        bash scripts/stop_db.sh
    

Connecting to Rafiki's Redis
--------------------------------------------------------------------

You can connect to Redis DB with `rebrow <https://github.com/marians/rebrow>`_:

    .. code-block:: shell

        bash scripts/start_rebrow.sh

...with these credentials by default:

    ::

        RAFIKI_ADDR=127.0.0.1
        REDIS_EXT_PORT=6380

Pushing Images to Docker Hub
--------------------------------------------------------------------

To push the Rafiki's latest images to Docker Hub (e.g. to reflect the latest code changes):

    .. code-block:: shell

        bash scripts/push_images.sh

Building Rafiki's Documentation
--------------------------------------------------------------------

Rafiki uses `Sphinx documentation <http://www.sphinx-doc.org>`_ and hosts the documentation with `Github Pages <https://pages.github.com/>`_ on the `gh-pages branch <https://github.com/nginyc/rafiki/tree/gh-pages>`_. 
Build & view Rafiki's Sphinx documentation on your machine with the following commands:

    .. code-block:: shell

        bash scripts/build_docs.sh latest
        open docs/index.html

Testing
--------------------------------------------------------------------

Rafiki uses `pytest <https://docs.pytest.org>`_.  

Running all tests:

    ::

        pip install -r rafiki/requirements.txt
        pip install -r rafiki/advisor/requirements.txt
        pip install -r test/requirements.txt
        bash scripts/test.sh


Troubleshooting
--------------------------------------------------------------------

While building Rafiki's images locally, if you encounter errors like "No space left on device", 
you might be running out of space allocated for Docker. Try one of the following:

    ::

        # Prunes dangling images
        docker system prune --all

    ::

        # Delete all containers
        docker rm $(docker ps -a -q)
        # Delete all images
        docker rmi $(docker images -q)

From Mac Mojave onwards, due to Mac's new `privacy protection feature <https://www.howtogeek.com/361707/how-macos-mojaves-privacy-protection-works/>`_, 
you might need to explicitly give Docker *Full Disk Access*, restart Docker, or even do a factory reset of Docker.


Using Rafiki Admin's HTTP interface
--------------------------------------------------------------------

To make calls to the HTTP endpoints of Rafiki Admin, you'll need first authenticate with email & password 
against the `POST /tokens` endpoint to obtain an authentication token `token`, 
and subsequently add the `Authorization` header for every other call:

::

    Authorization: Bearer {{token}}