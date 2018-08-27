import os

class DatabaseConfig(object):
    host = os.environ['POSTGRES_HOST'] or 'localhost'
    port = os.environ['POSTGRES_PORT'] or 5432
    user = os.environ['POSTGRES_USER'] or 'rafiki'
    db = os.environ['POSTGRES_DB'] or 'rafiki'
    password = os.environ['POSTGRES_PASSWORD'] or 'rafiki'