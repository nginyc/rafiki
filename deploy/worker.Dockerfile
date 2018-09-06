FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY src/worker/requirements.txt worker/requirements.txt
RUN pip install -r worker/requirements.txt

COPY src/db/ db/
COPY src/common/ common/
COPY src/model/ model/
COPY src/worker/ worker/

# Copy init script
COPY scripts/start_worker.py start_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

ENTRYPOINT [ "python", "start_worker.py" ]