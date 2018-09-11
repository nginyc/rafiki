
from common import ServiceType
from train_worker import TrainWorker

# Inference
RUNNING_INFERENCE_WORKERS = 'running__inference_workers'
REQUEST_QUEUE = 'request_queue'
QFE_SLEEP = 0.25
INFERENCE_WORKER_SLEEP = 0.25
BATCH_SIZE = 32

# Worker
SERVICE_TYPE_TO_WORKER_CLASS = {
    ServiceType.TRAIN: TrainWorker
}