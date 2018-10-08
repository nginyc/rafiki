import os

from rafiki.utils.log import configure_logging
from rafiki.predictor.app import app

service_id = os.environ['RAFIKI_SERVICE_ID']
container_id = os.environ.get('HOSTNAME', 'localhost')

configure_logging('service-{}-{}'.format(service_id, container_id))

if __name__ == "__main__":
    app.run(host='0.0.0.0', 
            port=os.getenv('PREDICTOR_PORT', 8002), 
            debug=True, 
            threaded=True)
