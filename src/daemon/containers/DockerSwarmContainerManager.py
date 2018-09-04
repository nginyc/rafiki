import abc
import os
import docker

from .ContainerManager import ContainerManager

class DockerSwarmContainerManager(ContainerManager):
    def __init__(self,
        network=os.environ.get('DOCKER_NETWORK', 'rafiki')):
        self._network = network
        self._client = docker.from_env()

    def create_service(self, service_name, image_name, replicas, args, environment_vars):
        env = [
            '{}={}'.format(k, v)
            for (k, v) in environment_vars.items()
        ]

        service = self._client.services.create(
            image=image_name,
            args=args,
            networks=[self._network],
            name=service_name,
            env=env
        )
        service.scale(replicas)
        service_id = service.id
        return service_id

    def update_service(self, service_id, replicas):
        service = self._client.services.get(service_id)
        service.scale(replicas)

    def destroy_service(self, service_id):
        service = self._client.services.get(service_id)
        service.remove()
        
        