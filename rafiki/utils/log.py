import os
import logging
import tempfile
from datetime import datetime
import json
import traceback
import time

logger = logging.getLogger(__name__)

def configure_logging(process_name):
    # Configure all logging to a log file
    logs_folder_path = os.environ['LOGS_DOCKER_WORKDIR_PATH']
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='{}/{}.log'.format(logs_folder_path, process_name))
