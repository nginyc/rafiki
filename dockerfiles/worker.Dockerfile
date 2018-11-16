FROM nvidia/cuda:9.0-base-ubuntu16.04

RUN apt-get update && apt-get -y upgrade

# `tensorflow-gpu` dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cuda-command-line-tools-9-0 \
      cuda-cublas-9-0 \
      cuda-cufft-9-0 \
      cuda-curand-9-0 \
      cuda-cusolver-9-0 \
      cuda-cusparse-9-0 \
      libcudnn7=7.2.1.38-1+cuda9.0 \
      libnccl2=2.2.13-1+cuda9.0 \
      libfreetype6-dev \
      libhdf5-serial-dev \
      libpng12-dev \
      libzmq3-dev \
      pkg-config \
      software-properties-common \
      unzip \
      && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
    apt-get update && \
    apt-get install libnvinfer4=4.1.2-1+cuda9.0

# Install conda with pip and python 3.6
RUN apt-get -y install curl bzip2 \
  && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
  && bash /tmp/miniconda.sh -bfp /usr/local \
  && rm -rf /tmp/miniconda.sh \
  && conda create -y --name rafiki python=3.6 \
  && conda clean --all --yes
ENV PATH /usr/local/envs/rafiki/bin:$PATH
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH $DOCKER_WORKDIR_PATH
RUN pip install --upgrade pip

ARG DOCKER_WORKDIR_PATH

RUN mkdir $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH

# Install python dependencies
COPY rafiki/utils/requirements.txt utils/requirements.txt
RUN pip install -r utils/requirements.txt
COPY rafiki/db/requirements.txt db/requirements.txt
RUN pip install -r db/requirements.txt
COPY rafiki/cache/requirements.txt cache/requirements.txt
RUN pip install -r cache/requirements.txt
COPY rafiki/model/requirements.txt model/requirements.txt
RUN pip install -r model/requirements.txt
COPY rafiki/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt
COPY rafiki/worker/requirements.txt worker/requirements.txt
RUN pip install -r worker/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

CMD ["bash", "-c", "source activate rafiki; $WORKER_INSTALL_COMMAND; python scripts/start_worker.py"]