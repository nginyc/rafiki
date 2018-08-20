# API Documentation

## Roles

- App User
- App Developer
- Model Contributor
- Rafiki Developer

## Classes

### App

- constructor(name: String, train_dataset: Dataset, test_dataset: Dataset, task: Task)
- train(budget: TrainJobBudget): TrainJob
- deploy(train_job: TrainJob): DeploymentJob
- id: String
- name: String
- task: Task
- models: [TrainedModel]
- model_deployments: [DeploymentJob]

### Dataset

- url: String
- input_shape: [Int]
- output_shape: [Int]

### DeploymentJob

- id: String
- train_job: TrainJob
- status: Enum('deployed', 'deploying')

### TrainJob

- id: String
- app: App
- version: Int
- budget: TrainJobBudget
- status: Enum('completed', 'in_progress')

### Model (Defined by Model Contributor)

- constructor(hyperparameters_set: HyperparameterSet)
- train(dataset: Dataset)
- dump_parameters(): ModelParameterSet
- load_parameters(model_params: ModelParameterSet)
- predict(queries: [DatasetQuery]): [DatasetLabel]
- evaluate(dataset: Dataset): [ModelMetric]
- hyperparameter_set: HyperparameterSet
- hyperparameter_set_config: HyperparameterSetConfig

### TrainedModel extends Model

- id: String
- train_job: TrainJob
- test_metrics: [ModelMetric]
- task: Task

### TrainJobBudget

- type: Enum('classifier', 'minutes')
- amount: int

### ModelMetric

- type: String
- value: Double

### ModelParameterSet

Dict<parameter_name: String, paramater_value: Bytes>

### Task

String

### DatasetQuery

Any

### DatasetLabel

Int

### HyperparameterSet

Dict<hyperparameter_name: String, hyperparameter_value: String>

### HyperparameterSetConfig

Similar to https://github.com/HDI-Project/ATM/blob/master/docs/source/add_method.rst
 
