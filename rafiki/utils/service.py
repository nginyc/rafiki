#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import os
import signal
import traceback
import logging

from rafiki.utils.log import configure_logging

logger = logging.getLogger(__name__)

def run_worker(meta_store, start_worker, stop_worker):
    service_id = os.environ['RAFIKI_SERVICE_ID']
    service_type = os.environ['RAFIKI_SERVICE_TYPE']
    container_id = os.environ.get('HOSTNAME', 'localhost')
    configure_logging('service-{}-worker-{}'.format(service_id, container_id))

    def _sigterm_handler(_signo, _stack_frame):
        logger.warn("Terminal signal received: %s, %s" % (_signo, _stack_frame))
        stop_worker()
        exit(0)

    signal.signal(signal.SIGINT, _sigterm_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Mark service as running in DB
    with meta_store:
        service = meta_store.get_service(service_id)
        meta_store.mark_service_as_running(service)

    try:
        logger.info('Starting worker "{}" for service of ID "{}"...'.format(container_id, service_id))
        start_worker(service_id, service_type, container_id)
        logger.info('Stopping worker...')
        stop_worker()

    except Exception as e:
        logger.error('Error while running worker:')
        logger.error(traceback.format_exc())

        # Mark service as errored in DB
        with meta_store:
            service = meta_store.get_service(service_id)
            meta_store.mark_service_as_errored(service)

        stop_worker()

        raise e

    
    



