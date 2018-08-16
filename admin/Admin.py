import pandas as pd

import atm
from atm.enter_data import enter_data
from atm.database import Database
from atm.predict import predict, get_dataset_example

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

    def create_datarun(self, dataset_name, preparator_type, 
        preparator_params, budget_type, budget):
        run_config = self._build_run_config(
            dataset_name=dataset_name,
            preparator_type=preparator_type,
            preparator_params=preparator_params,
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

    def get_dataset(self, dataset_id):
        dataset = self._db.get_dataset(dataset_id)
        
        if not dataset:
            raise Exception('No such dataset')

        return {
            'id': dataset.id,
            'name': dataset.name,
            'preparator_type': dataset.preparator_type,
            'preparator_params': dataset.preparator_params,
            'n_examples': dataset.n_examples,
            'k_classes': dataset.k_classes,
            'd_features': dataset.d_features
        }
        

    def get_datarun(self, datarun_id):
        datarun = self._db.get_datarun(datarun_id)

        if not datarun:
            raise Exception('No such datarun')

        classifier = self._db.get_best_classifier(
            score_target='cv',  # TODO: change to accuracy on test data
            datarun_id=datarun_id
        )

        return {
            'id': datarun_id,
            'status': datarun.status,
            'budget': datarun.budget,
            'budget_type': datarun.budget_type,
            'start_time': datarun.start_time,
            'end_time': datarun.end_time,
            'dataset_id': datarun.dataset_id,
            'best_classifier_id': classifier.id if classifier else None
        }


    def get_dataset_example(self, dataset_id, example_id):
        query, label = get_dataset_example(self._db, dataset_id, example_id)

        return {
            'query': query,
            'label': label
        }


    def get_classifier(self, classifier_id):
        classifier = self._db.get_classifier(classifier_id)
        hyperpartition = self._db.get_hyperpartition(
            classifier.hyperpartition_id)
        return {
            'id': classifier_id,
            'datarun_id': classifier.datarun_id,
            'method': hyperpartition.method,
            'hyperparameters': classifier.hyperparameter_values,
            'cv_accuracy': float(classifier.cv_judgment_metric)
        }

    def query_classifier(self, classifier_id, queries):
        predictions = predict(self._db, classifier_id, queries, self._log_config)
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

    def _build_run_config(self, dataset_name, preparator_type, 
            preparator_params, budget_type, budget):
        x = atm.config.RunConfig()
        x.dataset_name = dataset_name
        x.preparator_type = preparator_type
        x.preparator_params = preparator_params
        x.methods = ['one_layer_tf']
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
