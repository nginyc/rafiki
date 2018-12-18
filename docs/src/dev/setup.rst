.. _`setup-configuration`:

Setup & Configuration
====================================================================

.. contents:: Table of Contents


.. _`quick-setup`:

Quick Setup
--------------------------------------------------------------------

We assume development or deployment in a MacOS or Linux environment.

1. Install Docker 18 & Python 3.6

2. Clone the project at https://github.com/nginyc/rafiki

3. Setup Rafiki's complete stack with the init script:

    .. code-block:: shell

        bash scripts/start.sh

*Rafiki Admin* and *Rafiki Admin Web* will be available at ``127.0.0.1:3000`` and ``127.0.0.1:3001`` respectively.

To destroy Rafiki's complete stack:

    .. code-block:: shell

        bash scripts/stop.sh

Adding Nodes to Rafiki
--------------------------------------------------------------------

Rafiki's default setup runs on a single node. 

Rafiki has with its dynamic stack (e.g. train workers, inference workes, predictors) 
running as `Docker Swarm Services <https://docs.docker.com/engine/swarm/services/>`_. 
It runs in a `Docker routing-mesh overlay network <https://docs.docker.com/network/overlay/>`_,
using it for networking amongst nodes.

To scale Rafiki horizontally, add more nodes to the master node's Docker Swarm:

1. If Rafiki is running, stop Rafiki, and have the master node leave its Docker Swarm

2. Put every worker node and a master node into a common network,
   and change ``DOCKER_SWARM_ADVERTISE_ADDR`` in ``.env.sh`` to the IP address of the master node
   in *the network that your worker nodes are in*

3. For every node, including the master node, ensure the `firewall rules 
   allow TCP & UDP traffic on ports 2377, 7946 and 4789 
   <https://docs.docker.com/network/overlay/#operations-for-all-overlay-networks>`_ 

4. Start Rafiki with `bash scripts/start.sh`

5. For every worker node, have the node `join the master node's Docker Swarm <https://docs.docker.com/engine/swarm/join-nodes/>`_

6. On the *master* node, for *every* node, configure it with the script:

    ::    

        bash scripts/setup_node.sh


Enabling GPU for Rafiki's Workers
--------------------------------------------------------------------

Rafiki's default setup would only configure its workers to run on CPUs across Rafiki's nodes. 

Rafiki's workers run in Docker containers that extend the Docker image ``nvidia/cuda:9.0-runtime-ubuntu16.04``,
and are capable of leveraging on `CUBA-Capable GPUs <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions>`_
across Rafiki's nodes. 

To allow model training in workers to run on GPUs, perform the following configuration on *each* node in Rafiki:

1. `Install NVIDIA drivers <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ for CUDA *9.0* or above

2. `Install nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`_

3. Set the ``default-runtime`` of Docker to `nvidia` 
   (e.g. `instructions here <https://lukeyeager.github.io/2018/01/22/setting-the-default-docker-runtime-to-nvidia.html>`_)


Exposing Rafiki Publicly
--------------------------------------------------------------------

Rafiki Admin and Rafiki Admin Web runs on the master node. 
Change ``RAFIKI_ADDR`` in ``.env.sh`` to the IP address of the master node
in the network you intend to expose Rafiki in.

Example: 

::

    export RAFIKI_ADDR=172.28.176.35

Re-deploy Rafiki. Rafiki Admin and Rafiki Admin Web will be available at that IP address,
over ports 3000 and 3001 (by default), assuming incoming connections to these ports are allowed.


Reading Rafiki's logs
--------------------------------------------------------------------

By default, you can read logs of Rafiki Admin, Rafiki Advisor & any of Rafiki's services
in `./logs` directory at the root of the project's directory of the master node. 


Troubleshooting
--------------------------------------------------------------------

Q: There seems to be connectivity issues amongst containers across nodes!

A: Ensure that containers are able to communicate with one another through the Docker Swarm overlay network:
   https://docs.docker.com/network/network-tutorial-overlay/#use-an-overlay-network-for-standalone-containers
