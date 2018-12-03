import os
from rafiki.utils.service import run_service
from rafiki.db import Database
from rafiki.constants import ServiceType

worker = None

def start_service(service_id, service_type):
    global worker

    # Run install command
    install_command = os.environ.get('WORKER_INSTALL_COMMAND', '')
    exit_code = os.system(install_command)
    if exit_code != 0: 
        # TODO: Fix failing install command for `pip install torch==0.4.1;``
        raise Exception('Install command gave non-zero exit code: "{}"'.format(install_command))

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

def end_service(service_id, service_type):
    global worker
    if worker is not None:
        worker.stop()    

db = Database()
run_service(db, start_service, end_service)
