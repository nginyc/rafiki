FROM python:3.6

RUN apt-get update

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
COPY rafiki/predictor/requirements.txt predictor/requirements.txt
RUN pip install -r predictor/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

EXPOSE 8002

ENTRYPOINT [ "python", "scripts/start_predictor.py" ]