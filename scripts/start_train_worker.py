import sys
import logging
import os

from train_worker import TrainWorker

if len(sys.argv) < 2:
    print('Usage: python {} <worker_id>'.format(__file__))
    exit(1)

worker_id = sys.argv[1]

# Configure all logging to a log file
LOGS_FOLDER_PATH = os.environ['LOGS_FOLDER_PATH']
CONTAINER_ID = os.environ['HOSTNAME']
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='{}/worker-{}-{}.log'.format(LOGS_FOLDER_PATH, worker_id, CONTAINER_ID))

worker = TrainWorker(worker_id)
worker.start()