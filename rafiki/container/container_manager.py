import abc
import os
from typing import List, Dict, Tuple

class InvalidServiceRequestError(Exception): pass

class ContainerService():
    def __init__(self, id: str, hostname: str, port: int, info: Dict[str, any] = {}):
        self.id = id # ID for the service created
        self.hostname = hostname # Hostname for the service created (in the internal network)
        self.port = port # Port for the service created (in the internal network), None if no container port is passed
        self.info = info

class ContainerManager(abc.ABC):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_service(self, service_name: str, docker_image: str, args: List[str], environment_vars: Dict[str, str], 
                        mounts: Dict[str, str] = {}, replicas: int = 1, publish_port: List[Tuple[int, int]] = None, 
                        gpus: int = 0) -> ContainerService:
        '''
            Creates a service with a set number of replicas. Replicas will be created *on the same node*.
            The service should regenerate replicas if they exit with a non-zero code. 
            However, if a replica exits with code 0, it should not regenerate the replica.
   
            :param str service_name: Name of the service
            :param str docker_image: Name of the Docker image for the container of the service
            :param int replicas: No. of replicas for the service
            :param List[str] args: Command-line arguments to pass to each replica
            :param Dict[str, str] environment_vars: Dict of environment variable names to values to pass to each replica
            :param Dict[str, str] mounts: Dict of host directory to container directory for mounting of volumes on each replica
            :param List[Tuple[int, int]] publish_port: (<published_host_port>, <container_port>) 
            :param int gpus: No. of GPUs to exclusively allocate to this service (shared across replicas)
            :rtype: ContainerService
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy_service(self, service: ContainerService):
        '''
            Stops & destroys a service
            
            :param ContainerService service: Container service to destroy
        '''
        raise NotImplementedError()