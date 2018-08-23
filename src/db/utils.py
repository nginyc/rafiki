import uuid
import datetime

def generate_uuid():
    return str(uuid.uuid4())

def generate_datetime():
    return datetime.datetime.utcnow()