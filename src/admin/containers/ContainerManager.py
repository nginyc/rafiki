import abc
import os

class ContainerManager(abc.ABC):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_service(self, dockerfile_path, name, replicas):
        '''
            Creates a service 
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def update_service(self, name, replicas):
        '''
            Updates the service's properties e.g. scaling the number of replicas
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy_service(self, name):
        '''
            Stops & destroys a service
        '''
        raise NotImplementedError()
