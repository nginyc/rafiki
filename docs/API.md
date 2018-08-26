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
    1. [Application Developer](#application_developer)
    2. [Model Developer](#model_developer)
    3. [Application User](#application_user)
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
    library/
        __init__.py
        application.py
        dataset.py
        model.py
        task.py
        model/
            __init__.py
            convolution_neural_network.py
            multi_layer_perceptron.py
        task/
            __task__.py
            image_classification.py
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

    def start_rafiki(self, 
                     database_image, 
                     frontend_image, 
                     redis_image):
        """ Starts a new Rafiki cluster.
        
        Parameters
        ----------
        database_image: str {optional}
            The database docker image to use. You can set this argument
            to specify a custom build of the database.
        frontend_image: str {optional}
            The frontend docker image to use. You can set this argument
            to specify a custom build of the frontend.
        redis_image: str {optional}
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

    def create_application(self, 
                           name, 
                           version, 
                           task_name, 
                           models, 
                           train_dataset_zip_file, 
                           test_dataset_zip_file):
        """ Creates an application.
        
        Parameters
        ----------
        name: str
            Name of application. It must be unique.
        version: str
            Version of application.
        task_name: str
            The name of the machine learning task that the application
            is built for. The task name must be an element of the list
            of task names returned by `RafikiConnection.get_task_names`.
        models: list(str)
            The list of model names to be used by Rafiki. All model 
            names must stem from the list of model names returned by
            `RafikiConnection.get_task_models`.
        train_dataset_zip_file: str
            The path to the train dataset zip file. The format of the
            zip file must follow the specifications defined by the 
            `Task`.
        test_dataset_zip_file: str
            The path to the test dataset zip file. The format of the zip
            file must follow the specifications defined by the `Task`.

        Returns
        -------
        `Application`:
            An `Application` instance that was created.

        Raises
        ------
        `RafikiException`
        """
        pass

    def create_model(self, model, tasks):
        """ Creates a model.

        Parameters
        ----------
        tasks: list(str)
            The list of task names that the model would be categorised
            under. All task names must stem from the list returned by 
            `RafikiConnectionn.get_task_names`.
        model: `BaseModel`
            An instance of a concrete subclass of `BaseModel`.

        Raises
        ------
        `RafikiException`
        """
        pass
    
    def create_task(self, name):
        """
        """
        pass

    def get_task_names(self):
        """
        """
        pass
    
    def get_application(self, name):
        """
        """
        pass

    def get_application_names(self):
        """
        """
        pass

    def get_task_models(self, task):
        """
        """
        pass
```

<a name="roles"></a>
## Roles

<a name="application_developer"></a>
### Application Developer

``` python
def main():
    rafiki = RafikiConnection(KubernetesContainerManager())
    rafiki.connect() # or rafiki.start_rafiki()
    task_names = rafiki.get_task_names()

    if ("image-classification" not in set(task_names)):
        return
    
    app = rafiki.create_application(name="food-identifier", 
                                    version="v1", 
                                    task_name="image-classification"
                                    train_dataset_zip_file="./train.zip",
                                    test_dataset_zip_file="./test.zip")
    train_job = app.train(budget_type="trial_count", budget_amount="10")
    
    while not train_job.has_completed():
        pass
    
    trained_model_names = app.get_trained_model_names(sort="DESC")
    app.deploy_model(name=trained_models[0])
    app.deploy_model(name=trained_models[1])
    app.deploy_model(name=trained_models[2])
    app.predict(input=image) # image must be in bytes or string
```

<a name="model_developer"></a>
### Model_Developer

``` python
from rafiki_admin.library.model import Model

class BayesianClassifier(Model): pass # Assumes abstract methods are implemented.

def main():
    rafiki = RafikiConnection(KubernetesContainerManager())
    rafiki.connect() # or rafiki.start_rafiki()   
```

<a name="application_user"></a>
### Application User

<a name="schemas"></a>
## Schemas