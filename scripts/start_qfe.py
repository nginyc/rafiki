import os
import logging

from qfe import app

# Configure all logging to a log file
LOGS_FOLDER_PATH = os.environ['LOGS_FOLDER_PATH']
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='{}/admin.log'.format(LOGS_FOLDER_PATH))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv('QFE_PORT', 8001))