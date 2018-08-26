# API Documentation

### Table of Contents

1. [Packages Overview](#packages_overview)
    1. [Client Package](#client_package)
        * [rafiki_connection.py](#client_rafiki_connection)
        * [container_manager.py](#client_container_manager)
    2. [Library Package](#library_package)
        * [application.py](#library_application)
        * [model.py](#library_model)
        * [task.py](#library_task)
        * [train_job.py](#library_train_job)
        * [worker.py](#library_worker)
2. [Roles](#roles)
    1. [Application Developer](#application_developer)
    2. [Model Developer](#model_developer)
        * [Defining new Model](#defining_new_model)
        * [Defining new Task](#defining_new_task)
    3. [Application User](#application_user)
3. [Schemas](#schemas)
    1. [Application](#schema_application)
    2. [Model](#schema_model)
    3. [Task](#schema_task)
    4. [Train Job](#schema_train_job)
    5. [Trial](#schema_trial)
    6. [Application Trial Junction](#schema_application_trial)
    7. [Task Model Junction](#schema_task_model)

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
        model.py
        task.py
        train_job.py
        worker.py
        metric.py
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
        container_manager: `ContainerManager`
            An instance of a concrete subclass of `ContainerManager`.
        """
        pass

    def start_rafiki(self, database_image, frontend_image, redis_image):
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
                           model_names, 
                           train_dataset_zip_file, 
                           test_dataset_zip_file):
        """ Creates a new `Application` object.
        
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
        model_names: list(str)
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

    def create_model(self, model, task_names):
        """ Creates a new model.

        Parameters
        ----------
        model: `Model`
            An instance of a concrete subclass of `Model`.
        task_names: list(str)
            The list of task names that the model would be categorised
            under. All task names must stem from the list returned by 
            `RafikiConnectionn.get_task_names`.

        Raises
        ------
        `RafikiException`
        """
        pass
    
    def create_task(self, task):
        """ Creates a new machine learning task.

        Parameters
        ----------
        task: `Task`
            An instance of a concrete subclass of `Task`.

        Raises
        ------
        `RafikiException`
        """
        pass

    def get_task_names(self):
        """ Returns list of supported machine learning task.

        Returns
        -------
        list(str):
            A list of task names.
        """
        pass
    
    def get_application(self, name, version):
        """ Returns instance of an `Application`.

        Parameters
        ----------
        name: str
            Name of application.
        version: str
            Version of application.

        Returns
        -------
        `Application`:
            An `Application` instance that was created.     

        Raises
        ------
        `RafikiException`  
        """
        pass

    def get_application_names_and_versions(self):
        """ Returns list of applications' name and version.

        Returns
        -------
        list(tuple(str,str)):
            A list containing (application_name, application_version).
        """
        pass

    def get_task_model_names(self, task_name):
        """ Returns all model names associated with the task name.

        Returns
        -------
        list(str):
            A list of model names.
        """
        pass
```

<a name="client_container_manager"></a>
**container_manager.py**

``` python
from abc import ABC, abstractmethod

class ContainerManager(ABC):

    @abstractmethod
    def start_rafiki(self, databse_image, frontend_image, redis_image):
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

    @abstractmethod
    def connect(self):
        """ Connects to an existing Rafiki cluser. 
        
        Raises
        ------
        `RafikiException`
        """
        pass
```

<a name="library_package"></a>
### Library Package

<a name="library_application"></a>
**application.py**

``` python
class Application:

    def __init__(self, name, version, model_names, task_name):
        """ Creates an instance of `Application`. """
        pass

    def train(self, budget_type, budget_amount):
        """ Starts a train job.

        Parameters
        ----------
        budget_type: str
            The type of budget.
        budget_amount: str
            The total resource allocated to the training job.
        
        Returns
        -------
        `TrainJob`:
            An instance of `TrainJob`.
        """
        pass

    def evaluate(self):
        """ Evaluate deployed models.

        Returns
        -------
        `Metric`:
            An instance of `Metric`.
        """
        pass

    def predict(self, input):
        """ Returns prediction from deployed models.

        Parameters
        ----------
        input: [str]/[int]/[double]/[bytes]
            The input for prediction.

        Returns
        -------
        str:
            The predicted result.
        """
        pass

    def get_trained_model_names(sort):
        pass

```

<a name="library_model"></a>
**model.py**

``` python
from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_name(self):
        """ Returns model name.

        Returns
        -------
        str:
            Name of model.
        """
        pass

    @abstractmethod
    def get_hyperparameters_config(self):
        """ Returns dictionary of conditional parameter tree.

        Returns
        -------
        dict:
            Dictionary of conditional parameter tree.
        """
        pass
    
    @abstractmethod
    def init(self, hyperparameters):
        """ True init method.

        Parameters
        ----------
        hyperparameters: dict
            Selected hyperparameters and their values.
        """
        pass

    @abstractmethod
    def preprocess(self, train_dataset_features):
        """ Do preprocessing before train, evaluate and predict methods.

        Parameters
        ----------
        train_dataset_features: nd-list
            N-dimension array of features from train dataset.
        """
        pass

    @abstractmethod
    def train(self, train_dataset_features, train_dataset_labels):
        """ Train method.

        Parameters
        ----------
        train_dataset_features: nd-list
            N-dimension array of features from train dataset.
        train_dataset_labels: list
            Labels of train dataset.
        """
        pass

    @abstractmethod
    def evaluate(self, test_dataset_features, test_dataset_labels):
        """ Evaluate method.

        Parameters
        ----------
        test_dataset_features: nd-list
            N-dimension array of features from test dataset.
        test_dataset_labels: list
            Labels of test dataset.   
        
        Returns
        -------
        `Metric`:
            An instance of `Metric` object.
        """
        pass

    @abstractmethod
    def predict(self, inputs):  
        """ Predict method.

        Parameters
        ----------
        inputs: [str]/[int]/[double]/[bytes]
            The input for prediction.

        Returns
        -------
        str:
            The predicted result.
        """
        pass

    @abstractmethod
    def load_parameters(self, parameters):
        """ Initialised model with parameters.

        Parameters
        ----------
        parameters: dict
            Parameters to initalise model.
        """
        pass

    @abstractmethod
    def dump_parameters(self):
        """ Returns dictionary of parameters to save.

        Returns
        -------
        dict:
            Dictionary of parameters to save.
        """
        pass

    @abstractmethod
    def destroy(self):
        """ Cleaning up method. """
        pass

```

<a name="library_task"></a>
**task.py**

``` python
from abc import ABC, abstractmethod

class Task(ABC):

    def __init__(self, name, train_dataset_zip_file, test_dataset_zip_file):
        self.name = name
        self.train_dataset_zip_file = train_dataset_zip_file
        self.test_dataset_zip_file = test_dataset_zip_file

    @abstractmethod
    def get_train_dataset_features(self):
        pass

    @abstractmethod
    def get_train_dataset_labels():
        pass

    @abstractmethod
    def get_test_dataset_features():
        pass

    @abstractmethod
    def get_test_dataset_labels():
        pass

```

<a name="library_train_job"></a>
**train_job.py**

``` python
from enum import Enum

class Status(Enum):
    CREATED = "CREATED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TrainJob:
    def __init__(budget_type, budget_amount):
        self.status = Status.CREATED

    def has_completed():
        return self.status == Status.COMPLETED

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

    model_names = rafiki.get_task_model_names(task_name="image-classification")
    if not (set(["cnn", "mlp"]) < set(model_names)):
        return
    
    app = rafiki.create_application(name="food-identifier", 
                                    version="v1", 
                                    task_name="image-classification",
                                    model_names=["cnn, mlp"],
                                    train_dataset_zip_file="./train.zip",
                                    test_dataset_zip_file="./test.zip")
    train_job = app.train(budget_type="trial_count", budget_amount="10")
    
    while not train_job.has_completed():
        pass
    
    trained_model_names = app.get_trained_model_names(sort="DESC")
    app.deploy_model(name=trained_models[0])
    app.deploy_model(name=trained_models[1])
    app.deploy_model(name=trained_models[2])
```

<a name="model_developer"></a>
### Model_Developer

<a name="defining_new_model"></a>
**Defining new Model**

``` python
from rafiki_admin.library.model import Model

# Assumes abstract methods are implemented.
class BayesianNetwork(Model): pass 

def main():
    rafiki = RafikiConnection(KubernetesContainerManager())
    rafiki.connect() # or rafiki.start_rafiki()  
 
    task_names = rafiki.get_task_names()
    if ("image-classification" not in set(task_names)):
        return

    bayesian_network = BatesianNetwork()
    rafiki.create_model(model=bayesian_network, 
                        task_names=["image_classification"])
```

<a name="defining_new_task"></a>
**Defining new Task**

``` python
from rafiki_admin.library.task import Task

# Assumes abstract methods are implemented.
class NLP(Task): pass 

def main():
    rafiki = RafikiConnection(KubernetesContainerManager())
    rafiki.connect() # or rafiki.start_rafiki()  
    nlp = NLP()
    rafiki.create_task(task=nlp)
```

<a name="application_user"></a>
### Application User

``` python
import base64

def main():
    with open("image.png", "rb") as image:
        image_str = base64.b64encode(image.read())

    rafiki = RafikiConnection(KubernetesContainerManager())
    rafiki.connect() # or rafiki.start_rafiki()  
    app = rafiki.get_application(name="food-identifier", version="v1")
    result = app.predict(input=[image_str])
```

<a name="schemas"></a>
## Schemas

<a name="schema_application"></a>
### Application

``` SQL
CREATE TABLE Application (
    Name varchar(255) NOT NULL,
    Version varchar(255) NOT NULL,
    TaskName varchar(255),
    TrainJobID int,
    PRIMARY KEY (Name, Version),
    FOREIGN KEY (TaskName) REFERENCES Task(Name),
    FOREIGN KEY (TrainJobID) REFERENCES TrainJob(ID)
);
```

<a name="schema_model"></a>
### Model

``` SQL
CREATE TABLE Model (
    Name varchar(255) PRIMARY KEY,
    ModelSerialised binary NOT NULL
);
```

<a name="schema_task"></a>
### Task

``` SQL
CREATE TABLE Task (
      Name varchar(255) PRIMARY KEY,
      TaskSerialised binary NOT NULL
);
```

<a name="schema_train_job"></a>
### Train Job

``` SQL
CREATE TABLE TrainJob (
    ID int PRIMARY KEY,
    BudgetType varchar(255) NOT NULL,
    BudgetAmount varchar(255) NOT NULL,
    Status varchar(255) NOT NULL,
    ApplicationName varchar(255),
    FOREIGN KEY (ApplicationName) REFERENCES Application(Name)
)
```

<a name="schema_trial"></a>
### Trial

``` SQL
CREATE TABLE Trial (
    ID int PRIMARY KEY,
    Score varchar(255),
    TrainJobID int,
    FOREIGN KEY (TrainJobID) REFERENCES TrainJob(ID)
);
```

<a name="schema_application_trial"></a>
### Application Trial Junction

Represents deployed models.

``` SQL
CREATE TABLE ApplicationTrialJunction (
    ApplicationName varchar(255) NOT NULL,
    ApplicationVersion varchar(255) NOT NULL,
    TrialID int NOT NULL,
    PRIMARY KEY (ApplicationName, ApplicationVersion, TrialID),
    FOREIGN KEY (ApplicationName) REFERENCES Application(Name),
    FOREIGN KEY (ApplicationVersion) REFERENCES Application(Version),
    FOREIGN KEY (TrialID) REFERENCES Trial(ID)
);
```

<a name="schema_task_model"></a>
### Task Model Junction

Represents available models associated with the task.

``` SQL
CREATE TABLE TaskModelJunction(
    TaskName varchar(255) NOT NULL,
    ModelName varchar(255) NOT NULL,
    PRIMARY KEY (TaskName, ModelName),
    FOREIGN KEY (TaskName) REFERENCES Task(Name),
    FOREIGN KEY (ModelName) REFERENCES Model(Name)
);
```