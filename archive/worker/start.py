import os

from .Worker import Worker

def start():
  worker = Worker(
    host=os.environ['MYSQL_HOST'],
    port=os.environ['MYSQL_PORT'],
    password=os.environ['MYSQL_PASSWORD'],
    username=os.environ['MYSQL_USER'],
    database=os.environ['MYSQL_DATABASE']
  )

  worker.start()