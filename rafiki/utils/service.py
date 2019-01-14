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

def run_service(db, start_service, end_service):
    service_id = os.environ['RAFIKI_SERVICE_ID']
    service_type = os.environ['RAFIKI_SERVICE_TYPE']
    container_id = os.environ.get('HOSTNAME', 'localhost')
    configure_logging('service-id-{}-c-{}'.format(service_id, container_id))

    def _sigterm_handler(_signo, _stack_frame):
        logger.warn("Terminal signal received: %s, %s" % (_signo, _stack_frame))

        # Mark service as stopped in DB
        with db:
            service = db.get_service(service_id)
            db.mark_service_as_stopped(service)

        end_service(service_id, service_type)
        exit(0)

    signal.signal(signal.SIGINT, _sigterm_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Mark service as running in DB
    with db:
        service = db.get_service(service_id)
        db.mark_service_as_running(service)

    try:
        logger.info('Starting service {}...'.format(service_id))

        start_service(service_id, service_type)

        logger.info('Ending service {}...'.format(service_id))

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

    
    



