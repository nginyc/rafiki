import os
import logging

from rafiki.utils.service import run_worker
from rafiki.meta_store import MetaStore
from rafiki.predictor.predictor import Predictor
from rafiki.predictor.app import app

logger = logging.getLogger(__name__)

global_predictor: Predictor = None

def start_worker(service_id, service_type, container_id):
    global global_predictor
    
    logger.info('Starting global predictor...')
    global_predictor = Predictor(service_id)
    global_predictor.start()

    app.run(host='0.0.0.0', 
            port=os.getenv('PREDICTOR_PORT', 3003), 
            threaded=True)

def stop_worker():
    global global_predictor

    if global_predictor is not None:
        global_predictor.stop()
        global_predictor = None

meta_store = MetaStore()
run_worker(meta_store, start_worker, stop_worker)
