
.. _`installing-kubernetes`:

Installing Kubernetes
====================================================================

Usage of Rafiki in Kubernetes mode requires Kubernetes 1.15+. 

To achieve this, we recommend the instructions below:

    1. Install kubelet kubeadm kubectl

        .. code-block:: shell

            apt-get install -y kubelet kubeadm kubectl --allow-unauthenticated

    2. Close swap

        .. code-block:: shell

            swapoff -a

    3. Config cri and change docker mode to systemd, reference to `Kubernetes Container runtimes <https://kubernetes.io/docs/setup/production-environment/container-runtimes/>`__

    4. Edit /etc/default/kubelet

        Environment=
        KUBELET_EXTRA_ARGS=--cgroup-driver=systemd

    5. Reset kubeadm, maybe not necessary
        
        .. code-block:: shell

            kubeadm reset

    6. Init k8s service, use your own host ip and the node name you want
        
        .. code-block:: shell
            
            kubeadm init --kubernetes-version=1.15.1 --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=YOURHOSTIP --node-name=YOURNODENAME --ignore-preflight-errors=ImagePull

    7. Add Kubernetes config to current user

        .. code-block:: shell

            mkdir -p $HOME/.kube
            sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
            sudo chown $(id -u):$(id -g) $HOME/.kube/config

    8. If just a single node, set master node as worker node

        .. code-block:: shell
            
            kubectl taint nodes --all node-role.kubernetes.io/master-

    9. Install flannel from github

        .. code-block:: shell

            kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml

    10. Config role

        .. code-block:: shell

            kubectl create clusterrolebinding add-on-cluster-admin --clusterrole=cluster-admin --serviceaccount=default:default

    11. Nodeport range setting

        .. code-block:: shell
            
            sudo vim /etc/kubernetes/manifests/kube-apiserver.yaml

        set "- --service-node-port-range=1-65535" in spec.containers.command node

Otherwise, you can refer to these links below on installing Kubernetes: 

    - `Kubernetes Setup Documentation <https://kubernetes.io/docs/setup/>`_
    