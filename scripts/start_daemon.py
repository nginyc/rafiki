import logging
import os

from daemon import Daemon

# Configure all logging to a log file
LOGS_FOLDER_PATH = os.environ['LOGS_FOLDER_PATH']
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='{}/daemon.log'.format(LOGS_FOLDER_PATH))

daemon = Daemon()
daemon.start()