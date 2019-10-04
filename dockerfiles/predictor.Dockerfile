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

FROM ubuntu:16.04

RUN apt-get update && apt-get -y upgrade

# Install conda with pip and python 3.6
ARG CONDA_ENVIORNMENT
RUN apt-get -y install curl bzip2 \
  && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
  && bash /tmp/miniconda.sh -bfp /usr/local \
  && rm -rf /tmp/miniconda.sh \
  && conda create -y --name $CONDA_ENVIORNMENT python=3.6 \
  && conda clean --all --yes
ENV PATH /usr/local/envs/$CONDA_ENVIORNMENT/bin:$PATH
RUN pip install --upgrade pip
ENV PYTHONUNBUFFERED 1

ARG DOCKER_WORKDIR_PATH
RUN mkdir -p $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

# Install python dependencies
COPY singaauto/requirements.txt singaauto/requirements.txt
RUN pip install -r singaauto/requirements.txt
COPY singaauto/utils/requirements.txt singaauto/utils/requirements.txt
RUN pip install -r singaauto/utils/requirements.txt
COPY singaauto/meta_store/requirements.txt singaauto/meta_store/requirements.txt
RUN pip install -r singaauto/meta_store/requirements.txt
COPY singaauto/redis/requirements.txt singaauto/redis/requirements.txt
RUN pip install -r singaauto/redis/requirements.txt
COPY singaauto/kafka/requirements.txt singaauto/kafka/requirements.txt
RUN pip install -r singaauto/kafka/requirements.txt
COPY singaauto/predictor/requirements.txt singaauto/predictor/requirements.txt
RUN pip install -r singaauto/predictor/requirements.txt

COPY singaauto/ singaauto/
COPY scripts/ scripts/

EXPOSE 3003

CMD ["python", "scripts/start_predictor.py"]