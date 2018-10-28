import os
import signal
import traceback
import logging

from rafiki.utils.log import configure_logging

logger = logging.getLogger(__name__)

def run_service(db, start_service, end_service):
    service_id = os.environ['RAFIKI_SERVICE_ID']
    service_type = os.environ['RAFIKI_SERVICE_TYPE']
    container_id = os.environ.get('HOSTNAME', 'localhost')
    configure_logging('service-{}-{}'.format(service_id, container_id))

    service = db.get_service(service_id)

    def _sigterm_handler(_signo, _stack_frame):
        print("SIGTERM received: %s, %s" % (_signo, _stack_frame))

        # Mark service as errored in DB
        with db:
            service = db.get_service(service_id)
            db.mark_service_as_errored(service)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print('Starting service {}...'.format(service_id))

    # Mark service as running in DB
    with db:
        service = db.get_service(service_id)
        db.mark_service_as_running(service)

    try:
        start_service(service_id, service_type)

        # Mark service as stopped in DB
        with db:
            service = db.get_service(service_id)
            db.mark_service_as_stopped(service)

        end_service(service_id, service_type)

    except Exception as e:
        logger.error('Error while running service:')
        logger.error(traceback.format_exc())

        # Mark service as errored in DB
        with db:
            service = db.get_service(service_id)
            db.mark_service_as_errored(service)

        end_service(service_id, service_type)

        raise e

    
    



