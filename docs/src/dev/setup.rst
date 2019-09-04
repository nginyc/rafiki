.. _`setup-configuration`:

Setup & Configuration
====================================================================

.. _`quick-setup`:

Quick Setup
--------------------------------------------------------------------

We assume development or deployment in a MacOS or Linux environment.

1. Install Docker 18 (`Ubuntu <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`__, `MacOS <https://docs.docker.com/docker-for-mac/install/>`__)
   and, if required, add your user to ``docker`` group (`Linux <https://docs.docker.com/install/linux/linux-postinstall/>`__).

.. note::

    If you're not a user in the ``docker`` group, you'll instead need ``sudo`` access and prefix every bash command with ``sudo -E``.

2. Install Kubernetes 1.15+ (see :ref:`installing-kubernetes`) if using Kubernetes.

3. Install Python 3.6 such that the ``python`` and ``pip`` commands point to the correct installation of Python 3.6 (see :ref:`installing-python`).

4. Clone the project at https://github.com/nginyc/rafiki (e.g. with `Git <https://git-scm.com/downloads>`__)

5. If using docker, Setup Rafiki's complete stack with the setup script:

    .. code-block:: shell

        bash scripts/start.sh

   If using kubernetes, Setup Rafiki's complete stack with the setup script:

    .. code-block:: shell
        
        bash scripts/kubernetes/start.sh

*Rafiki Admin* and *Rafiki Web Admin* will be available at ``127.0.0.1:3000`` and ``127.0.0.1:3001`` respectively.

If using docker, to destroy Rafiki's complete stack:

    .. code-block:: shell

        bash scripts/stop.sh
        
If using kubernetes, to destroy Rafiki's complete stack:
        
    .. code-block:: shell

        bash scripts/kubernetes/stop.sh

Scaling Rafiki
--------------------------------------------------------------------

Rafiki's default setup runs on a single machine and only runs its workloads on CPUs.

Rafiki's model training workers run in Docker containers that extend the Docker image ``nvidia/cuda:9.0-runtime-ubuntu16.04``,
and are capable of leveraging on `CUDA-Capable GPUs <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions>`__

Scaling Rafiki horizontally and enabling GPU usage involves setting up *Network File System* (*NFS*) at a common path across all nodes,
installing & configuring the default Docker runtime to `nvidia` for each GPU-bearing node. If using docker swarm, putting all these nodes into a single Docker Swarm.
If using kubernetes, putting all these nodes into kubernetes.

.. seealso:: :ref:`architecture`

To run Rafiki on multiple machines with GPUs on docker swarm, do the following:

1. If Rafiki is running, stop Rafiki with ``bash scripts/stop.sh``

2. Have all nodes `leave any Docker Swarm <https://docs.docker.com/engine/reference/commandline/swarm_leave/>`__ they are in

3. Set up NFS such that the *master node is a NFS host*, *other nodes are NFS clients*, and the master node *shares an ancestor directory 
   containing Rafiki's project directory*. `Here are instructions for Ubuntu <https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-16-04>`__

4. All nodes should be in a common network. On the *master node*, change ``DOCKER_SWARM_ADVERTISE_ADDR`` in the project's ``.env.sh`` to the IP address of the master node
   in *the network that your nodes are in*

5. For *each node* (including the master node), ensure the `firewall rules 
   allow TCP & UDP traffic on ports 2377, 7946 and 4789 
   <https://docs.docker.com/network/overlay/#operations-for-all-overlay-networks>`_

6. For *each node that has GPUs*:

    6.1. `Install NVIDIA drivers <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__ for CUDA *9.0* or above

    6.2. `Install nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__
    
    6.3. Set the ``default-runtime`` of Docker to `nvidia` (e.g. `instructions here <https://lukeyeager.github.io/2018/01/22/setting-the-default-docker-runtime-to-nvidia.html>`__)

7. On the *master node*, start Rafiki with ``bash scripts/start.sh``

8. For *each worker node*, have the node `join the master node's Docker Swarm <https://docs.docker.com/engine/swarm/join-nodes/>`__

9. On the *master* node, for *each node* (including the master node), configure it with the script:

    ::    

        bash scripts/setup_node.sh

To run Rafiki on multiple machines with GPUs on kubernetes, do the following:

1. If Rafiki is running, stop Rafiki with ``bash scripts/kubernetes/stop.sh``

2. Put all nodes you need in kubernetes cluster, reference to `kubeadm join <https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm-join/>`__

3. Set up NFS such that the *master node is a NFS host*, *other nodes are NFS clients*, and the master node *shares an ancestor directory 
   containing Rafiki's project directory*. `Here are instructions for Ubuntu <https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-16-04>`__

4. Change ``KUBERNETES_ADVERTISE_ADDR`` in the project's ``scripts/kubernetes/.env.sh`` to the IP address of the master node
   in *the network that your nodes are in*

5. For *each node that has GPUs*:

    6.1. `Install NVIDIA drivers <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__ for CUDA *9.0* or above

    6.2. `Install nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__
    
    6.3. Set the ``default-runtime`` of Docker to `nvidia` (e.g. `instructions here <https://lukeyeager.github.io/2018/01/22/setting-the-default-docker-runtime-to-nvidia.html>`__)

    6.4. Install nvidia-device-plugin, use command "*kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.10/nvidia-device-plugin.yml*" on the *master node*

7. On the *master node*, start Rafiki with ``bash scripts/kubernetes/start.sh``

Exposing Rafiki Publicly
--------------------------------------------------------------------

Rafiki Admin and Rafiki Web Admin runs on the master node. 
If using docker swarm, change ``RAFIKI_ADDR`` in ``.env.sh`` to the IP address of the master node
in the network you intend to expose Rafiki in.
If using kubernetes, change ``RAFIKI_ADDR`` in ``scripts/kubernetes/.env.sh`` to the IP address of the master node
in the network you intend to expose Rafiki in.

Example: 

::

    export RAFIKI_ADDR=172.28.176.35

Re-deploy Rafiki. Rafiki Admin and Rafiki Web Admin will be available at that IP address,
over ports 3000 and 3001 (by default), assuming incoming connections to these ports are allowed.

**Before you expose Rafiki to the public, 
it is highly recommended to change the master passwords for superadmin, server and the database (located in `.env.sh` as `POSTGRES_PASSWORD`, `APP_SECRET` & `SUPERADMIN_PASSWORD`)**

Reading Rafiki's logs
--------------------------------------------------------------------

By default, you can read logs of Rafiki Admin & any of Rafiki's workers
in ``./logs`` directory at the root of the project's directory of the master node. 


Troubleshooting
--------------------------------------------------------------------

Q: There seems to be connectivity issues amongst containers across nodes!

A: `Ensure that containers are able to communicate with one another through the Docker Swarm overlay network <https://docs.docker.com/network/network-tutorial-overlay/#use-an-overlay-network-for-standalone-containers>`__
