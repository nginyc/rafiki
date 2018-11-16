import os

from rafiki.utils.service import run_service
from rafiki.db import Database
from rafiki.predictor.app import app

def start_service(service_id, service_type):
    app.run(host='0.0.0.0', 
            port=os.getenv('PREDICTOR_PORT', 8002), 
            threaded=True)

def end_service(service_id, service_type):
    pass

db = Database()
run_service(db, start_service, end_service)
