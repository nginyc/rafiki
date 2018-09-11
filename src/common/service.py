class TrainJobStatus():
    STARTED = 'STARTED'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'

class TrialStatus():
    STARTED = 'STARTED'
    ERRORED = 'ERRORED'
    COMPLETED = 'COMPLETED'

class WorkerStatus():
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class ServiceStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class ServiceType():
    TRAIN = 'TRAIN'
    INFERENCE = 'INFERENCE'