FROM python:3.6

RUN apt-get update

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY src/qfe/requirements.txt qfe/requirements.txt
RUN pip install -r qfe/requirements.txt

COPY src/qfe/ qfe/
COPY src/cache cache/
COPY src/__init__.py __init__.py 
COPY src/config.py config.py

# Copy init script
COPY scripts/start_qfe.py start_qfe.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

EXPOSE 8001

CMD ["python", "start_qfe.py"]