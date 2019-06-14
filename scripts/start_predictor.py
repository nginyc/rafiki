import os

from rafiki.utils.service import run_worker
from rafiki.db import Database
from rafiki.predictor.app import app

def start_worker(service_id, service_type, container_id):
    app.run(host='0.0.0.0', 
            port=os.getenv('PREDICTOR_PORT', 3003), 
            threaded=True)

def stop_worker():
    pass

db = Database()
run_worker(db, start_worker, stop_worker)
