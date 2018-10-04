FROM python:3.6

RUN apt-get update

RUN mkdir /root/rafiki/
WORKDIR /root/rafiki/

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

# Copy init script
COPY scripts/start_predictor.py start_predictor.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

EXPOSE 8002

ENTRYPOINT [ "python", "start_predictor.py" ]