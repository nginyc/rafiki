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
import logging
import tempfile
from datetime import datetime
import json
import traceback
import time

logger = logging.getLogger(__name__)

def configure_logging(process_name):
    # Configure all logging to a log file
    logs_folder_path = os.path.join(os.environ['WORKDIR_PATH'], os.environ['LOGS_DIR_PATH'])
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)s %(levelname)s %(message)s', 
                    filename='{}/{}.log'.format(logs_folder_path, process_name))
