FROM ubuntu:16.04

RUN apt-get update && apt-get -y upgrade

# Install conda with pip and python 3.6
RUN apt-get -y install curl bzip2 \
  && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
  && bash /tmp/miniconda.sh -bfp /usr/local \
  && rm -rf /tmp/miniconda.sh \
  && conda create -y --name rafiki python=3.6 \
  && conda clean --all --yes
ENV PATH /usr/local/envs/rafiki/bin:$PATH
RUN pip install --upgrade pip
ENV PYTHONUNBUFFERED 1

ARG DOCKER_WORKDIR_PATH
RUN mkdir $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

# Install python dependencies
COPY rafiki/utils/requirements.txt utils/requirements.txt
RUN pip install -r utils/requirements.txt
COPY rafiki/db/requirements.txt db/requirements.txt
RUN pip install -r db/requirements.txt
COPY rafiki/model/requirements.txt model/requirements.txt
RUN pip install -r model/requirements.txt
COPY rafiki/container/requirements.txt container/requirements.txt
RUN pip install -r container/requirements.txt
COPY rafiki/admin/requirements.txt admin/requirements.txt
RUN pip install -r admin/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

EXPOSE 3000

CMD ["bash", "-c", "source activate rafiki; python scripts/start_admin.py"]
