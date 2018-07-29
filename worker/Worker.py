import time
import os

import atm
from atm.worker import work
from atm.database import Database

class Worker:
  def __init__(self, host, port, username, password, database):
    log_config = self._build_log_config()
    atm.config.initialize_logging(log_config)
    self._sql_config = self._build_sql_config(
      host, port, username, password, database
    )

  def start(self):
    db = Database(
      **vars(self._sql_config)
    )

    work(
      db,
      save_files=True
    )

  def _build_log_config(self):
    x = atm.config.LogConfig()
    x.model_dir = os.path.join(os.getcwd(), './models/') 
    return x

  def _build_sql_config(self, host, port, username, password, database):
    x = atm.config.SQLConfig()
    x.dialect = 'mysql'
    x.database = database
    x.username = username
    x.host = host
    x.port = port
    x.password = password
    return x