FROM python:3.6

RUN apt-get update

RUN mkdir /root/rafiki/
WORKDIR /root/rafiki/

# Install python dependencies
COPY rafiki/utils/requirements.txt utils/requirements.txt
RUN pip install -r utils/requirements.txt
COPY rafiki/db/requirements.txt db/requirements.txt
RUN pip install -r db/requirements.txt
COPY rafiki/query_frontend/requirements.txt query_frontend/requirements.txt
RUN pip install -r query_frontend/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_query_frontend.py start_query_frontend.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

EXPOSE 8002

ENTRYPOINT [ "python", "start_query_frontend.py" ]