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

FROM node:11.1-alpine

ARG DOCKER_WORKDIR_PATH
RUN mkdir -p $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH

# Inject the following commands on docker run
# ENV RAFIKI_ADDR=ncrs.d2.comp.nus.edu.sg
# ENV ADMIN_EXT_PORT=7500
# ENV WEB_ADMIN_EXT_PORT=7501

COPY web/package.json web/package.json
COPY web/yarn.lock web/yarn.lock

RUN cd web/ && yarn install --production

COPY web/ web/

EXPOSE 3001

CMD cd web/ && yarn build && node app.js
