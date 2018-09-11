import sys
import logging
import os

from worker import start_worker

if len(sys.argv) < 2:
    print('Usage: python {} <service_id>'.format(__file__))
    exit(1)

service_id = sys.argv[1]
container_id = os.environ.get('HOSTNAME', 'localhost')

# Configure all logging to a log file
LOGS_FOLDER_PATH = os.environ['LOGS_FOLDER_PATH']
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='{}/worker-{}-{}.log'.format(LOGS_FOLDER_PATH, service_id, container_id))

start_worker(container_id, service_id)