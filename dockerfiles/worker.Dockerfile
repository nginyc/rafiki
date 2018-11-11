FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

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

# Install popular ML libraries
RUN pip install numpy==1.14.5 tensorflow==1.10.1 h5py==2.8.0 torch==0.4.1 Keras==2.2.2 scikit-learn==0.20.0

COPY rafiki/ rafiki/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

ENTRYPOINT [ "python", "scripts/start_worker.py" ]