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
COPY rafiki/model/requirements.txt model/requirements.txt
RUN pip install -r model/requirements.txt
COPY rafiki/container/requirements.txt container/requirements.txt
RUN pip install -r container/requirements.txt
COPY rafiki/admin/requirements.txt admin/requirements.txt
RUN pip install -r admin/requirements.txt

COPY rafiki/ rafiki/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

EXPOSE 8000

CMD ["python", "scripts/start_admin.py"]