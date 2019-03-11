import os
import signal
import traceback
import logging

from rafiki.utils.log import configure_logging

logger = logging.getLogger(__name__)

def run_service(meta_store, start_service, end_service):
    service_id = os.environ['RAFIKI_SERVICE_ID']
    service_type = os.environ['RAFIKI_SERVICE_TYPE']
    container_id = os.environ.get('HOSTNAME', 'localhost')
    configure_logging('service-id-{}-c-{}'.format(service_id, container_id))

    def _sigterm_handler(_signo, _stack_frame):
        logger.warn("Terminal signal received: %s, %s" % (_signo, _stack_frame))
        end_service(service_id, service_type)
        exit(0)

    signal.signal(signal.SIGINT, _sigterm_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Mark service as running in DB
    with meta_store:
        service = meta_store.get_service(service_id)
        meta_store.mark_service_as_running(service)

    try:
        logger.info('Starting worker {}...'.format(service_id))

        start_service(service_id, service_type)

        logger.info('Ending worker {}...'.format(service_id))

        end_service(service_id, service_type)

    except Exception as e:
        logger.error('Error while running worker:')
        logger.error(traceback.format_exc())

        # Mark service as errored in DB
        with meta_store:
            service = meta_store.get_service(service_id)
            meta_store.mark_service_as_errored(service)

        end_service(service_id, service_type)

        raise e

    
    



