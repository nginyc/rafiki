FROM python:3.6

RUN mkdir /root/app/
WORKDIR /root/app/

# Install MySQL client
RUN apt-get update
RUN apt-get install -y mysql-client

ADD ./requirements.txt .

RUN pip install -r ./requirements.txt

EXPOSE 8000