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

ARG RAFIKI_IMAGE_WORKER
ARG RAFIKI_VERSION
FROM $RAFIKI_IMAGE_WORKER:$RAFIKI_VERSION

# Download file dependencies
COPY download_file_deps.sh download_file_deps.sh
COPY download_lm.sh download_lm.sh
COPY download_trie.sh download_trie.sh
COPY alphabet.txt alphabet.txt
RUN bash download_file_deps.sh