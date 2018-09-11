import abc
import os

class ContainerManager(abc.ABC):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_service(self, service_name, docker_image, 
        replicas, args, environment_vars, mounts={}):
        '''
            Creates a service with a set number of replicas.

            The service should regenerate replicas if they exit with a non-zero code. 
            However, if a replica exit with code 0, it should not regenerate the replica.

            Args
                service_name: String - Name of the service
                docker_image: String - Name of the Docker image to create a service for
                replicas: Int - Number of replicas to initialize for the service
                args: [String] - Arguments to pass to the service
                environment_vars: {String: String} - Dict of environment variable names to values
                mounts: {String: String} - Dict of host directory to container directory for mounting of volumes onto container

            Returns 
                id: String - ID for the service created
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def update_service(self, service_id, replicas):
        '''
            Updates the service's properties e.g. scaling the number of replicas

            Args
                service_id: String - ID of service to update
                replicas: Int - Adjusted number of replicas for the service
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy_service(self, service_id):
        '''
            Stops & destroys a service

            Args
                service_id: String - ID of service to destroy
        '''
        raise NotImplementedError()
