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
COPY rafiki/cache/requirements.txt cache/requirements.txt
RUN pip install -r cache/requirements.txt
COPY rafiki/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt
COPY rafiki/worker/requirements.txt worker/requirements.txt
RUN pip install -r worker/requirements.txt

# Install popular ML libraries
RUN pip install numpy==1.14.5 tensorflow==1.10.1 h5py==2.8.0 torch==0.4.1 Keras==2.2.2

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_worker.py start_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

ENTRYPOINT [ "python", "start_worker.py" ]