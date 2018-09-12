import sys
import os

from rafiki.constants import ServiceType
from rafiki.utils.log import configure_logging

service_id = os.environ['RAFIKI_SERVICE_ID']
service_type = os.environ['RAFIKI_SERVICE_TYPE']
container_id = os.environ.get('HOSTNAME', 'localhost')

configure_logging('service-{}-{}'.format(service_id, container_id))

if service_type == ServiceType.TRAIN:
    from rafiki.train_worker import TrainWorker
    worker = TrainWorker(service_id)
    worker.start()
else:
    raise Exception('Invalid service type: {}'.format(service_type))