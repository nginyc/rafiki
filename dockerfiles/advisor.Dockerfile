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
RUN mkdir $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

# Install python dependencies
COPY rafiki/utils/requirements.txt rafiki/utils/requirements.txt
RUN pip install -r rafiki/utils/requirements.txt
COPY rafiki/model/requirements.txt rafiki/model/requirements.txt
RUN pip install -r rafiki/model/requirements.txt
COPY rafiki/advisor/requirements.txt rafiki/advisor/requirements.txt
RUN pip install -r rafiki/advisor/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

EXPOSE 3002

CMD ["python", "scripts/start_advisor.py"]
