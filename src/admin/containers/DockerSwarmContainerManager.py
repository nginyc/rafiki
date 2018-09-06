import abc
import os
import docker
import logging

from .ContainerManager import ContainerManager

logger = logging.getLogger(__name__)

class DockerSwarmContainerManager(ContainerManager):
    def __init__(self,
        network=os.environ.get('DOCKER_NETWORK', 'rafiki')):
        self._network = network
        self._client = docker.from_env()

    def create_service(self, service_name, image_name, replicas, 
        args, environment_vars, mounts={}):
        env = [
            '{}={}'.format(k, v)
            for (k, v) in environment_vars.items()
        ]

        mounts_list = [
            '{}:{}:rw'.format(k, v)
            for (k, v) in mounts.items()
        ]

        service = self._client.services.create(
            image=image_name,
            args=args,
            networks=[self._network],
            name=service_name,
            env=env,
            mounts=mounts_list,
            # Restart replicas when they exit with error
            restart_policy={
                'Condition': 'on-failure'
            }
        )

        service.scale(replicas)
        service_id = service.id

        logger.info('Created service of ID {} (name: "{}") of {} x {} replicas' \
            .format(service_id, service_name, image_name, replicas))

        return service_id

    def update_service(self, service_id, replicas):
        service = self._client.services.get(service_id)
        service.scale(replicas)

        logger.info('Updated service of ID {} to {} replicas' \
            .format(service_id, replicas))


    def destroy_service(self, service_id):
        service = self._client.services.get(service_id)
        service.remove()

        logger.info('Deleted service of ID {}'.format(service_id))
        
        