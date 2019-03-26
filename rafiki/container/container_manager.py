import abc
import os
from enum import Enum
from collections import namedtuple
from typing import List, Dict

class InvalidServiceRequestError(Exception): pass

class ContainerService():
    def __init__(self, id: str, hostname: str, port: int, info: Dict[str, any] = {}):
        self.id = id # ID for the service created
        self.hostname = hostname # Hostname for the service created (in the internal network)
        self.port = port # Port for the service created (in the internal network), None if no container port is passed
        self.info = info

class ServiceRequirement(Enum):
    GPU = 'gpu' # Allocates a single GPU to the service

class ContainerManager(abc.ABC):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_service(self, service_name, docker_image, replicas, 
                        args, environment_vars, mounts={}, publish_port=None,
                        requirements: List[ServiceRequirement] = []) -> ContainerService:
        '''
            Creates a service with a set number of replicas.

            The service should regenerate replicas if they exit with a non-zero code. 
            However, if a replica exits with code 0, it should not regenerate the replica.

            Args
                service_name: String - Name of the service
                docker_image: String - Name of the Docker image to create a service for
                replicas: Int - Number of replicas to initialize for the service
                args: [String] - Arguments to pass to the service
                environment_vars: {String: String} - Dict of environment variable names to values
                mounts: {String: String} - Dict of host directory to container directory for mounting of volumes onto container
                publish_port: (<host_port>, <container_port>) - host port (port to be published) to container port 
                    The service should then be reachable at the host port on the host
                requirements: [ServiceRequirement] - List of requirements for the service
                
            Returns `Service`
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def update_service(self, service: ContainerService, replicas):
        '''
            Updates the service's properties e.g. scaling the number of replicas

            Args
                service: ContainerService to update
                replicas: Int - Adjusted number of replicas for the service
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy_service(self, service: ContainerService):
        '''
            Stops & destroys a service

            Args
                service: ContainerService to destroy
        '''
        raise NotImplementedError()
