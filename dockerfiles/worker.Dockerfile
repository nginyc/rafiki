FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY src/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt
COPY src/train_worker/requirements.txt train_worker/requirements.txt
RUN pip install -r train_worker/requirements.txt
COPY src/inference_worker/requirements.txt inference_worker/requirements.txt
RUN pip install -r inference_worker/requirements.txt

COPY src/cache/ cache/
COPY src/db/ db/
COPY src/common/ common/
COPY src/model/ model/
COPY src/worker/ worker/
COPY src/client/ client/
COPY src/__init__.py __init__.py 
COPY src/config.py config.py
COPY src/train_worker/ train_worker/
COPY src/inference_worker/ inference_worker/

# Copy init script
COPY scripts/start_worker.py scripts/start_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

ENTRYPOINT [ "python", "scripts/start_worker.py" ]
