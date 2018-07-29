import pandas as pd

import atm
from atm.enter_data import enter_data
from atm.database import Database

class Admin:
  def __init__(self, host, port, username, password, database):
    self._log_config = self._build_log_config()
    atm.config.initialize_logging(self._log_config)
    self._sql_config = self._build_sql_config(
      host, port, username, password, database
    )
    self._db = Database(
      **vars(self._sql_config)
    )

  def create_datarun(self, dataset_url, class_column, budget_type, budget):
    run_config = self._build_run_config(
      dataset_url=dataset_url, 
      class_column=class_column,
      budget_type=budget_type,
      budget=budget
    )
    id = enter_data(
      self._sql_config,
      run_config
    )
    return {
      'id': id
    }

  def get_datarun(self, datarun_id):
    datarun = self._db.get_datarun(datarun_id)
    classifier = self._db.get_best_classifier(
      score_target='cv', # TODO: change to accuracy on test data
      datarun_id=datarun_id
    )

    return {
      'id': datarun_id,
      'status': datarun.status,
      'budget': datarun.budget,
      'budget_type': datarun.budget_type,
      'start_time': datarun.start_time,
      'end_time': datarun.end_time,
      'best_classifier_id': classifier.id
    }

  def get_classifier(self, classifier_id):
    classifier = self._db.get_classifier(classifier_id)
    hyperpartition = self._db.get_hyperpartition(classifier.hyperpartition_id)
    return {
      'id': classifier_id,
      'method': hyperpartition.method,
      'hyperparameters': classifier.hyperparameter_values,
      'cv_accuracy': float(classifier.cv_judgment_metric)
    }

  def query_classifier(self, classifier_id, queries):
    model = self._db.load_model(classifier_id)
    query_df = pd.DataFrame(queries, index=range(len(queries)))
    predictions = model.predict(query_df)
    return {
      'queries': queries,
      'predictions': [x for x in predictions]
    }

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

  def _build_run_config(self, dataset_url, class_column, budget_type, budget):
    x = atm.config.RunConfig()
    x.train_path = dataset_url
    x.class_column = class_column
    x.methods = ['logreg', 'svm', 'sgd', 'dt', 'et', 'rf', 'gnb', 'mnb', 'bnb',
      'gp', 'pa', 'knn', 'mlp', 'ada']
    x.priority = 1
    x.budget_type = budget_type
    x.budget = budget
    x.tuner = 'uniform'
    x.selector = 'uniform'
    x.r_minimum = 2
    x.k_window = 3
    x.gridding = 0
    x.metric = 'f1'
    x.score_target = 'cv'
    return x
