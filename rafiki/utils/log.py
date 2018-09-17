import os
import logging

def configure_logging(process_name):
    # Configure all logging to a log file
    logs_folder_path = os.environ['LOGS_FOLDER_PATH']
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='{}/{}.log'.format(logs_folder_path, process_name))