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
COPY rafiki/utils/requirements.txt rafiki/utils/requirements.txt
RUN pip install -r rafiki/utils/requirements.txt
COPY rafiki/meta_store/requirements.txt rafiki/meta_store/requirements.txt
RUN pip install -r rafiki/meta_store/requirements.txt
COPY rafiki/model/requirements.txt rafiki/model/requirements.txt
RUN pip install -r rafiki/model/requirements.txt
COPY rafiki/client/requirements.txt rafiki/client/requirements.txt
RUN pip install -r rafiki/client/requirements.txt
COPY rafiki/advisor/requirements.txt rafiki/advisor/requirements.txt
RUN pip install -r rafiki/advisor/requirements.txt
COPY rafiki/param_store/requirements.txt rafiki/param_store/requirements.txt
RUN pip install -r rafiki/param_store/requirements.txt
COPY rafiki/cache/requirements.txt rafiki/cache/requirements.txt
RUN pip install -r rafiki/cache/requirements.txt
COPY rafiki/worker/requirements.txt rafiki/worker/requirements.txt
RUN pip install -r rafiki/worker/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

CMD ["python", "scripts/start_worker.py"]