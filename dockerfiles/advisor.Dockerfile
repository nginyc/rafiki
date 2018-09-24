FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/rafiki/
WORKDIR /root/rafiki/

# Install python dependencies
COPY rafiki/utils/requirements.txt utils/requirements.txt
RUN pip install -r utils/requirements.txt
COPY rafiki/advisor/requirements.txt advisor/requirements.txt
RUN pip install -r advisor/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_advisor.py start_advisor.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

EXPOSE 8000

CMD ["python", "start_advisor.py"]