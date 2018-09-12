FROM python:3.6

RUN apt-get update

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY rafiki/query_frontend/requirements.txt query_frontend/requirements.txt
RUN pip install -r query_frontend/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_query_frontend.py start_query_frontend.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

EXPOSE 8001

ENTRYPOINT [ "python", "start_query_frontend.py" ]