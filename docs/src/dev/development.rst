.. _`development`:

Development
====================================================================

**Before running any individual scripts, make sure to run the shell configuration script**:

    .. code-block:: shell

        source .env.sh

Refer to :ref:`architecture` and :ref:`folder-structure` for a developer's overview of SingaAuto.

.. _`testing-latest-code`:

Testing Latest Code Changes
--------------------------------------------------------------------

To test the lastet code changes e.g. in the ``dev`` branch, you'll need to do the following:

    1. Build SingaAuto's images on each participating node (the quickstart instructions pull pre-built `SingaAuto's images <https://hub.docker.com/r/singaautoai/>`_ from Docker Hub):

    .. code-block:: shell

        bash scripts/build_images.sh

    2. Purge all of SingaAuto's data (since there might be database schema changes):

    .. code-block:: shell

        bash scripts/clean.sh


Making a Release to ``master``
--------------------------------------------------------------------

In general, before making a release to ``master`` from ``dev``, ensure that the code at ``dev`` is stable & well-tested:
    
    1. Consider running all of SingaAuto's tests (see :ref:`testing-singaauto`). Remember to re-build the Docker images to ensure the latest code changes are reflected (see :ref:`testing-latest-code`)

    2. Consider running all of SingaAuto's example models in `./examples/models/ <https://github.com/nginyc/singaauto/tree/master/examples/models/>`_

    3. Consider running all of SingaAuto's example usage scripts in `./examples/scripts/ <https://github.com/nginyc/singaauto/tree/master/examples/scripts/>`_

    4. Consider running all of SingaAuto's example dataset-preparation scripts in `./examples/datasets/ <https://github.com/nginyc/singaauto/tree/master/examples/datasets/>`_

    5. Consider visiting SingaAuto Web Admin and manually testing it

    6. Consider building SingaAuto's documentation site and checking if the documentation matches the codebase (see :ref:`building-docs`)

After merging ``dev`` into ``master``, do the following:

    1. Build & push SingaAuto's new Docker images to `SingaAuto’s own Docker Hub account <https://hub.docker.com/u/singaautoai>`_:

        .. code-block:: shell

            bash scripts/build_images.sh
            bash scripts/push_images.sh

        Get Docker Hub credentials from @nginyc.

    2. Build & deploy SingaAuto's new documentation to ``SingaAuto's microsite powered by Github Pages``. Checkout SingaAuto's ``gh-pages`` branch, then run the following:

        .. code-block:: shell

            bash scripts/build_docs.sh latest

        Finally, commit all resultant generated documentation changes and push them to `gh-pages` branch. The latest documentation should be reflected at https://nginyc.github.io/singaauto/docs/latest/.
        
        Refer to `documentation on Github Pages <https://guides.github.com/features/pages/>` to understand more on how this works. 


    3. `Draft a new release on Github <https://github.com/nginyc/singaauto/releases/new>`_. Make sure to include the list of changes relative to the previous release.


Subsequently, you'll need to increase ``SINGAAUTO_VERSION`` in ``.env.sh`` to reflect a new release.


Managing SingaAuto's DB
--------------------------------------------------------------------

By default, you can connect to the PostgreSQL DB using a PostgreSQL client (e.g `Postico <https://eggerapps.at/postico/>`_) with these credentials:

    ::

        SINGAAUTO_ADDR=127.0.0.1
        POSTGRES_EXT_PORT=5433
        POSTGRES_USER=singaauto
        POSTGRES_DB=singaauto
        POSTGRES_PASSWORD=singaauto


You can start & stop SingaAuto's DB independently of the rest of SingaAuto's stack with:

    .. code-block:: shell

        bash scripts/start_db.sh
        bash scripts/stop_db.sh
    

Connecting to SingaAuto's Redis
--------------------------------------------------------------------

You can connect to Redis DB with `rebrow <https://github.com/marians/rebrow>`_:

    .. code-block:: shell

        bash scripts/start_rebrow.sh

...with these credentials by default:

    ::

        SINGAAUTO_ADDR=127.0.0.1
        REDIS_EXT_PORT=6380

Pushing Images to Docker Hub
--------------------------------------------------------------------

To push the SingaAuto's latest images to Docker Hub (e.g. to reflect the latest code changes):

    .. code-block:: shell

        bash scripts/push_images.sh

.. _`building-docs`:

Building SingaAuto's Documentation
--------------------------------------------------------------------

SingaAuto uses `Sphinx documentation <http://www.sphinx-doc.org>`_ and hosts the documentation with `Github Pages <https://pages.github.com/>`_ on the `gh-pages branch <https://github.com/nginyc/singaauto/tree/gh-pages>`_. 
Build & view SingaAuto's Sphinx documentation on your machine with the following commands:

    .. code-block:: shell

        bash scripts/build_docs.sh latest
        open docs/index.html

.. _`testing-singaauto`:

Running SingaAuto's Tests
--------------------------------------------------------------------

SingaAuto uses `pytest <https://docs.pytest.org>`_.  

First, start SingaAuto.

Then, run all integration tests with:

    ::

        pip install -r singaauto/requirements.txt
        pip install -r singaauto/advisor/requirements.txt
        pip install -r test/requirements.txt
        bash scripts/test.sh


Troubleshooting
--------------------------------------------------------------------

While building SingaAuto's images locally, if you encounter errors like "No space left on device", 
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


Using SingaAuto Admin's HTTP interface
--------------------------------------------------------------------

To make calls to the HTTP endpoints of SingaAuto Admin, you'll need first authenticate with email & password 
against the `POST /tokens` endpoint to obtain an authentication token `token`, 
and subsequently add the `Authorization` header for every other call:

::

    Authorization: Bearer {{token}}