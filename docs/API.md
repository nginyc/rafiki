# API Documentation

### Table of Contents

1. [Packages Overview](#packages_overview)
    1. [Client Package](#client_package)
        * [rafiki_connection.py](#client_rafiki_connection)
        * [container_manager.py](#client_container_manager)
    2. [Database Package](#database_package)
        * [database.py](#database_client_database)
        * [base.py](#database_schema_base)
2. [Roles](#roles)
3. [Schemas](#schemas)

<a name="packages_overview"></a>
## Packages Overview

``` bash
rafiki_admin/
    __init__.py
    client/
        __init__.py
        rafiki_connection.py
        container_manager.py
        kubernetes/
            __init__.py
            kubernetes_container_manager.py
            redis_service.yaml
            redis_deployment.yaml
            frontend_service.yaml
            frontend_deployment.yaml
            database_service.yaml
            database_deployment.yaml
        docker/
            __init__.py
            docker_container_manager.py
    class/
        __init__.py
        application.py
    database/
        __init__.py
        client/
            __init__.py
            database.py
            configuration.py
        schema/
            __init__.py
            base.py
            application.py
            dataset.py
            model.py
            task.py
            train_job.py
            trial.py 
    exception/
        __init__.py
        rafiki_exception.py    
```

<a name="client_package"></a>
### Client Package

<a name="client_rafiki_connection"></a>
**rafiki_connection.py**

``` python
class RafikiConnection(object):
    """ Main interface used to interact with Rafiki. """

    def __init__(self, container_manager):
        """ Creates a new `RafikiConnection` object.

        After creating a `RafikiConnection` instance, you still need to
        connect to a Rafiki cluster. You can connect to an existing
        cluster by calling `RafikiConnection.connect` or create a new 
        Rafiki cluser by calling `RafikiConnection.start_rafiki`.

        Parameters
        ----------
        container_manager: ContainerManager
            An instance of a concrete subclass of ContainerManager.
        """
        pass

    def start_rafiki(self, database_image, 
                     frontend_image, redis_image):
        """ Starts a new Rafiki cluster.
        
        Parameters
        ----------
        database_image: str(optional)
            The database docker image to use. You can set this argument
            to specify a custom build of the database.
        frontend_image: str(optional)
            The frontend docker image to use. You can set this argument
            to specify a custom build of the frontend.
        redis_image: str(optional)
            The redis docker image to use. You can set this argument
            to specify a custom build of the redis database.

        Raises
        ------
        `RafikiException`
        """
        pass

    def connect(self):
        """ Connects to an existing Rafiki cluser. 
        
        Raises
        ------
        `RafikiException`
        """
        pass

    def create_application(self, name, task, models, 
                           train_dataset_zip_file, 
                           test_dataset_zip_file):
        """ Creates an application

        Returns
        -------


        Raises
        ------
        `RafikiException`
        """
        pass

    def create_model(self, name, tasks, model):
        """

        tasks is list
        """
        pass
    
    def create_task(self, name):
        """
        """
        pass

    def add_model_to_tasks(self, name, tasks):
        """
        """
        pass

    def get_tasks(self):
        """
        """
        pass

    def get_applications(self):
        """
        """
        pass

    def get_models(self, task):
        """
        """
        pass
```

<a name="roles"></a>
## Roles

<a name="schemas"></a>
## Schemas