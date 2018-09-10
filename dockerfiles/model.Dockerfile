FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY src/model/requirements.txt model/requirements.txt
RUN pip install -r model/requirements.txt
COPY src/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt

COPY src/db/ db/
COPY src/common/ common/
COPY src/model/ model/
COPY src/train_worker/ train_worker/
COPY src/client/ client/

# Copy init script
COPY scripts/start_train_worker.py start_train_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

ENTRYPOINT [ "python", "start_train_worker.py" ]