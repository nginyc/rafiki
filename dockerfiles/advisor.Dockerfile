FROM nvidia/cuda:8.0-runtime-ubuntu16.04

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

ARG DOCKER_WORKDIR_PATH

RUN mkdir $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH

# Install python dependencies
COPY rafiki/utils/requirements.txt utils/requirements.txt
RUN pip install -r utils/requirements.txt
COPY rafiki/advisor/requirements.txt advisor/requirements.txt
RUN pip install -r advisor/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

EXPOSE 8001

CMD ["bash", "-c", "source activate rafiki; python scripts/start_advisor.py $@"]
