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
COPY rafiki/advisor/requirements.txt advisor/requirements.txt
RUN pip install -r advisor/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

EXPOSE 8001

CMD ["python", "scripts/start_advisor.py"]