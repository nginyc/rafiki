.. _`setup-configuration`:

Setup & Configuration
====================================================================

.. contents:: Table of Contents


.. _`quick-setup`:

Quick Setup
--------------------------------------------------------------------

We assume development or deployment in a MacOS or Linux environment.

1. Install Docker 18

2. Install Python 3.6

3. Setup Rafiki's complete stack with the init script:

.. code-block:: shell

    bash scripts/start.sh

*Rafiki Admin* and *Rafiki Admin Web* will be available at ``127.0.0.1:3000`` and ``127.0.0.1:3001`` respectively.

To destroy Rafiki's complete stack:

.. code-block:: shell

    bash scripts/stop.sh

Adding Nodes to Rafiki
--------------------------------------------------------------------

Rafiki has with its dynamic stack (e.g. train workers, inference workes, predictors) 
running as `Docker Swarm Services <https://docs.docker.com/engine/swarm/services/>`_.
Horizontal scaling can be done by `adding more nodes to the swarm <https://docs.docker.com/engine/swarm/join-nodes/>`_.

Exposing Rafiki Publicly
--------------------------------------------------------------------

Rafiki runs in a `Docker routing-mesh overlay network <https://docs.docker.com/network/overlay/>`_, with
Rafiki Admin and Rafiki Admin Web running only on the master node.

Edit the following line in ``.env.sh`` with the IP address of the master node in the network you intend to expose Rafiki:

.. code-block:: shell

    export RAFIKI_IP_ADDRESS=127.0.0.1

Re-deploy Rafiki. Rafiki Admin and Rafiki Admin Web will be available at that IP address over ports 3000 and 3001, 
assuming incoming connections to these ports are allowed.

Enabling GPU for Rafiki's Workers
--------------------------------------------------------------------

Rafiki's workers run in Docker containers that extend the Docker image ``nvidia/cuda:9.0-runtime-ubuntu16.04``,
and are capable of leveraging on `CUBA-Capable GPUs <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions>`_
across Rafiki's nodes. 

Rafiki's default setup would only configure its workers to run on CPUs across Rafiki's nodes. To allow model
training in workers to run on GPUs, perform the following configuration on *each* node in Rafiki:

1. `Install NVIDIA drivers <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ for CUDA *9.0* or above

2. `Install nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`_

3. Set the ``default-runtime`` of Docker to `nvidia` (e.g. `instructions here <https://lukeyeager.github.io/2018/01/22/setting-the-default-docker-runtime-to-nvidia.html>`_)

Reading Rafiki's logs
--------------------------------------------------------------------

You can read logs of Rafiki Admin, Rafiki Advisor & any of Rafiki's services at in the project's `./logs` directory.

Using Rafiki Admin's HTTP interface
--------------------------------------------------------------------

To make calls to the HTTP endpoints of Rafiki Admin, you'll need first authenticate with email & password 
against the `POST /tokens` endpoint to obtain an authentication token `token`, 
and subsequently add the `Authorization` header for every other call:

::

    Authorization: Bearer {{token}}
