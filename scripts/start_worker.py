import os
from rafiki.utils.service import run_worker
from rafiki.db import Database
from rafiki.constants import ServiceType

# Run install command
install_command = os.environ.get('WORKER_INSTALL_COMMAND', '')
exit_code = os.system(install_command)
if exit_code != 0: 
    raise Exception('Install command gave non-zero exit code: "{}"'.format(install_command))

worker = None

def start_worker(service_id, service_type, container_id):
    global worker

    if service_type == ServiceType.TRAIN:
        from rafiki.worker import TrainWorker
        worker = TrainWorker(service_id, container_id)
        worker.start()
    elif service_type == ServiceType.INFERENCE:
        from rafiki.worker import InferenceWorker
        worker = InferenceWorker(service_id)
        worker.start()
    else:
        raise Exception('Invalid service type: {}'.format(service_type))

def stop_worker():
    global worker
    if worker is not None:
        worker.stop()    

db = Database()
run_worker(db, start_worker, stop_worker)
