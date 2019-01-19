import os

from rafiki.utils.service import run_service
from rafiki.meta_store import MetaStore
from rafiki.predictor.app import app

def start_service(service_id, service_type):
    app.run(host='0.0.0.0', 
            port=os.getenv('PREDICTOR_PORT', 3003), 
            threaded=True)

def end_service(service_id, service_type):
    pass

meta_store = MetaStore()
run_service(meta_store, start_service, end_service)
