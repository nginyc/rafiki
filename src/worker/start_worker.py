import logging
import traceback

from db import Database
from train_worker import TrainWorker
from config import SERVICE_TYPE_TO_WORKER_CLASS

logger = logging.getLogger(__name__)

# Initializes an instance of a worker class and runs worker based on service type
def start_worker(container_worker_id, service_id):
    db = Database()

    with db:
        logger.info('Starting worker "{}" for service of id {}...' \
                    .format(container_worker_id, service_id))
        worker = db.create_worker(container_worker_id, service_id)
        db.commit()
        worker_id = worker.id

    try:
        with db:
            service = db.get_service(service_id)
            worker_class = SERVICE_TYPE_TO_WORKER_CLASS[service.service_type]
            worker_inst = worker_class(service.id, db)

        worker_inst.start()
        
        logger.info('Exiting worker with success...')
        
    except:
        logger.error('Error while running worker:')
        logger.error(traceback.format_exc())
        logger.error('Exiting worker with error...')

        with db:
            worker = db.get_worker(worker_id)
            db.mark_worker_as_errored(worker)

        # Exit with error
        exit(1)

    exit(0)