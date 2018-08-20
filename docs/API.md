# API Documentation

## Roles

- App User
- App Developer
- Model Contributor
- Rafiki Developer

## Classes

### App

- constructor(name: String, train_dataset: Dataset, test_dataset: Dataset, task: Task)
- get_trained_models(filter_by_metric: ModelMetric, sort_by_metric: ModelMetric): [TrainedModel]
- train(budget: TrainJobBudget): TrainJob
- deploy(model: TrainedModel): ModelDeployment

- id: String
- name: String
- task: Task
- models: [TrainedModel]
- model_deployments: [ModelDeployment]

### Dataset

- url: String

### ModelDeployment

- id: String
- model: TrainedModel
- status: Enum('running', 'deploying')

### TrainJob

- id: String
- app: App
- budget: TrainJobBudget
- status: Enum('completed', 'in_progress')

### Model (Defined by Model Contributor)

- constructor(hyperparameters_set: HyperparameterSet)
- preprocess(dataset: Dataset): [(DatasetQuery, DatasetLabel)]
- train([(DatasetQuery, DatasetLabel)])
- save_to_path(file_path: String)
- dump(): [Bytes]
- load_from_path(file_path: String)
- load(bytes: [Bytes])
- predict([DatasetQuery]): [DatasetLabel]
- evaluate([(DatasetQuery, DatasetLabel)]): [ModelMetric]

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

### DatasetQuery

Any

### DatasetLabel

Int

### Task

String

### HyperparameterSet

Dictionary(hyperparameter_name: String, hyperparameter_value: Double)

### HyperparameterSetConfig

Similar to https://github.com/HDI-Project/ATM/blob/master/docs/source/add_method.rst

#### Method

parse(json_file)

## Schema

 
