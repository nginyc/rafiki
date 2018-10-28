from rafiki.utils.service import run_service
from rafiki.db import Database
from rafiki.constants import ServiceType

worker = None

def start_service(service_id, service_type):
    if service_type == ServiceType.TRAIN:
        from rafiki.worker import TrainWorker
        worker = TrainWorker(service_id)
        worker.start()
    elif service_type == ServiceType.INFERENCE:
        from rafiki.worker import InferenceWorker
        worker = InferenceWorker(service_id)
        worker.start()
    else:
        raise Exception('Invalid service type: {}'.format(service_type))

def stop_service(service_id, service_type):
    if worker is not None:
        worker.stop()    

db = Database()
run_service(db, start_service, stop_service)
