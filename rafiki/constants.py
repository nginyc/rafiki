class BudgetType():
    MODEL_TRIAL_COUNT = 'MODEL_TRIAL_COUNT'
    ENABLE_GPU = 'ENABLE_GPU'

class ModelDependency():
    TENSORFLOW = 'tensorflow'
    KERAS = 'Keras'
    SCIKIT_LEARN = 'scikit-learn'
    PYTORCH = 'torch'
    SINGA = 'singa'
    numpy = 'numpy'
    scipy = 'scipy'
    Pillow = 'Pillow'
    Cython = 'Cython'
    matplotlib = 'matplotlib'
    opencv_python = 'opencv-python'
    imgaug = 'imgaug'
    pycocotools = 'rafiki-cocoapi'
    IPython = 'ipython'
    scikit_image = 'scikit-image'

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

class TrialStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    TERMINATED = 'TERMINATED'
    COMPLETED = 'COMPLETED'

class ServiceStatus():
    STARTED = 'STARTED'
    DEPLOYING = 'DEPLOYING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class ServiceType():
    TRAIN = 'TRAIN'
    PREDICT = 'PREDICT'
    INFERENCE = 'INFERENCE'

class UserType():
    SUPERADMIN = 'SUPERADMIN'
    ADMIN = 'ADMIN'
    MODEL_DEVELOPER = 'MODEL_DEVELOPER'
    APP_DEVELOPER = 'APP_DEVELOPER'
    USER = 'USER'

class AdvisorType():
    BTB_GP = 'BTB_GP'

class DatasetType():
    IMAGE_FILES = 'IMAGE_FILES'

class TaskType():
    IMAGE_CLASSIFICATION = 'IMAGE_CLASSIFICATION'
    POS_TAGGING = 'POS_TAGGING'
    OBJECT_DETECTION = 'OBJECT_DETECTION'
