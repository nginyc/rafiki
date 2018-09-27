FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/rafiki/
WORKDIR /root/rafiki/

# Install python dependencies
COPY rafiki/utils/requirements.txt utils/requirements.txt
RUN pip install -r utils/requirements.txt
COPY rafiki/db/requirements.txt db/requirements.txt
RUN pip install -r db/requirements.txt
COPY rafiki/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt
COPY rafiki/train_worker/requirements.txt train_worker/requirements.txt
RUN pip install -r train_worker/requirements.txt
COPY rafiki/inference_worker/requirements.txt inference_worker/requirements.txt
RUN pip install -r inference_worker/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_worker.py start_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

ENTRYPOINT [ "python", "start_worker.py" ]
