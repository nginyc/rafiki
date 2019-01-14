#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import abc
import os

class InvalidServiceRequest(Exception):
    pass

class ServiceRequirement():
    GPU = 'gpu'

class ContainerManager(abc.ABC):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_service(self, service_name, docker_image, replicas, 
                        args, environment_vars, mounts={}, publish_port=None,
                        requirements=[]):
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
                publish_port: (<host_port>, <container_port>) - host port (port to be published) to container port 
                    The service should then be reachable at the host port on the host
                requirements: [ServiceRequirement] - List of requirements for the service
                
            Returns {String: String} where
                id: String - ID for the service created
                hostname: String - Hostname for the service created (in the internal network)
                port: String - Port for the service created (in the internal network)
                    None if no container port is passed
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
