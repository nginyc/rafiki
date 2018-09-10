FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY src/admin/requirements.txt admin/requirements.txt
RUN pip install -r admin/requirements.txt

COPY src/db/ db/
COPY src/common/ common/
COPY src/model/ model/
COPY src/admin/ admin/
COPY src/__init__.py __init__.py 
COPY src/config.py config.py

# Copy init script
COPY scripts/start_admin.py start_admin.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

EXPOSE 8000

CMD ["python", "start_admin.py"]