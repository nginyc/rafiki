from typing import Dict

Budget = Dict[str, any]
ModelDependencies = Dict[str, str]

class BudgetOption():
    GPU_COUNT = 'GPU_COUNT'
    TIME_HOURS = 'TIME_HOURS'
    MODEL_TRIAL_COUNT = 'MODEL_TRIAL_COUNT'

class ModelAccessRight():
    PUBLIC = 'PUBLIC'
    PRIVATE = 'PRIVATE'

class InferenceJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class TrainJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERRORED = 'ERRORED'

class TrialStatus():
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    TERMINATED = 'TERMINATED'
    COMPLETED = 'COMPLETED'

class UserType():
    SUPERADMIN = 'SUPERADMIN'
    ADMIN = 'ADMIN'
    MODEL_DEVELOPER = 'MODEL_DEVELOPER'
    APP_DEVELOPER = 'APP_DEVELOPER'

class ServiceType():
    TRAIN = 'TRAIN'
    ADVISOR = 'ADVISOR'
    PREDICT = 'PREDICT'
    INFERENCE = 'INFERENCE'

class ServiceStatus():
    STARTED = 'STARTED'
    DEPLOYING = 'DEPLOYING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'
    
class ModelDependency():
    TENSORFLOW = 'tensorflow'
    KERAS = 'Keras'
    SCIKIT_LEARN = 'scikit-learn'
    TORCH = 'torch'
    TORCHVISION = 'torchvision'
    SINGA = 'singa'

class DatasetType():
    IMAGE_FILES = 'IMAGE_FILES'

class TaskType():
    IMAGE_CLASSIFICATION = 'IMAGE_CLASSIFICATION'
    POS_TAGGING = 'POS_TAGGING'