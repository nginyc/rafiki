import atm
from atm.enter_data import enter_data
from atm.database import Database

class Admin:
  def __init__(self, host, port, username, password, database):
    log_config = self._build_log_config()
    atm.config.initialize_logging(log_config)
    self._sql_config = self._build_sql_config(
      host, port, username, password, database
    )
    self._db = Database(
      **vars(self._sql_config)
    )

  def create_datarun(self, dataset_url, class_column):
    run_config = self._build_run_config(dataset_url, class_column)
    id = enter_data(
      self._sql_config,
      run_config
    )
    return {
      'id': id
    }

  def get_datarun(self, datarun_id):
    datarun = self._db.get_datarun(datarun_id)
    return datarun

  def _build_log_config(self):
    x = atm.config.LogConfig()
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

  def _build_run_config(self, dataset_url, class_column):
    x = atm.config.RunConfig()
    x.train_path = dataset_url
    x.class_column = class_column
    x.methods = ['logreg', 'dt', 'knn']
    x.priority = 1
    x.budget_type = 'classifier'
    x.budget = 100
    x.tuner = 'uniform'
    x.selector = 'uniform'
    x.r_minimum = 2
    x.k_window = 3
    x.gridding = 0
    x.metric = 'f1'
    x.score_target = 'cv'
    return x