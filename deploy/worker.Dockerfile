FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY src/worker/requirements.txt worker/requirements.txt
RUN pip install -r worker/requirements.txt
COPY src/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt
COPY src/inference_worker/requirments.txt inference_worker/requirments.txt
RUN pip install -r inference_worker/requirments.txt

COPY src/db/ db/
COPY src/common/ common/
COPY src/model/ model/
COPY src/worker/ worker/
COPY src/client/ client/
COPY src/cache cache/
COPY src/inference_worker inference_worker/

# Copy init script
COPY scripts/start_worker.py start_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

ENTRYPOINT [ "python", "start_worker.py" ]