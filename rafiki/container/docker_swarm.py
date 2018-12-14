import abc
import os
import time
import docker
import logging

from .container_manager import ContainerManager, ServiceRequirement, InvalidServiceRequest

logger = logging.getLogger(__name__)

class DockerSwarmContainerManager(ContainerManager):
    def __init__(self,
        network=os.environ.get('DOCKER_NETWORK', 'rafiki')):
        self._network = network
        self._client = docker.from_env()

    def create_service(self, service_name, docker_image, replicas, 
                        args, environment_vars, mounts={}, publish_port=None,
                        requirements=[]):
            
        env = [
            '{}={}'.format(k, v)
            for (k, v) in environment_vars.items()
        ]

        mounts_list = [
            '{}:{}:rw'.format(k, v)
            for (k, v) in mounts.items()
        ]

        ports_list = []
        container_port = None
        published_port = None
        hostname = service_name
        if publish_port is not None:
            # Host of Docker Swarm service = service's name at the container port
            published_port = int(publish_port[0])
            container_port = int(publish_port[1])
            ports_list = [{ 
                'PublishedPort': published_port, 
                'TargetPort': container_port
            }]

        # Gather list of constraints
        constraints = []
        if ServiceRequirement.GPU in requirements:
            # Check if there are nodes with GPU, raise error otherwise
            has_gpu = self._if_any_node_has_gpu()
            if not has_gpu:
                raise InvalidServiceRequest('There are no nodes with GPU to deploy the service on')

            constraints.append('node.labels.gpu!=0')
        
        service = self._client.services.create(
            image=docker_image,
            args=args,
            networks=[self._network],
            name=service_name,
            env=env,
            mounts=mounts_list,
            # Restart replicas when they exit with error
            restart_policy={
                'Condition': 'on-failure'
            },
            constraints=constraints,
            endpoint_spec={
                'Ports': ports_list
            },
            mode={
                'Replicated': {
                    'Replicas': replicas
                }
            }
        )

        service_id = service.id

        logger.info('Created service of ID {} (name: "{}") of {} x {} replicas' \
            .format(service_id, service_name, docker_image, replicas))

        return {
            'id': service_id,
            # Host of Docker Swarm service = service's name at the container port
            'hostname': hostname, 
            'port': container_port
        }

    def update_service(self, service_id, replicas):
        service = self._client.services.get(service_id)
        service.scale(replicas)

        logger.info('Updated service of ID {} to {} replicas' \
            .format(service_id, replicas))

    def destroy_service(self, service_id):
        service = self._client.services.get(service_id)
        service.remove()

        logger.info('Deleted service of ID {}'.format(service_id))
                
    def _if_any_node_has_gpu(self):
        nodes = self._client.nodes.list()
        has_gpu = False
        for node in nodes:
            gpu = int(node.attrs.get('Spec', {}).get('Labels', {}).get('gpu', 0))
            if gpu > 0:
                has_gpu = True

        return has_gpu