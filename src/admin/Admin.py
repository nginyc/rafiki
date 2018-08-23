

from db import Database, DatabaseConfig

from common import unserialize_model, serialize_model, BudgetConfig

class Admin(object):
    def __init__(self, database_config=DatabaseConfig()):
        self._db = Database(database_config)

    def create_app(self, name, task, train_dataset_config, test_dataset_config):
        with self._db:
            train_dataset = self._db.create_dataset(
                dataset_type=train_dataset_config.dataset_type,
                params=train_dataset_config.params
            )
            test_dataset = self._db.create_dataset(
                dataset_type=test_dataset_config.dataset_type,
                params=test_dataset_config.params
            )
            app = self._db.create_app(name, task, train_dataset, test_dataset)
            return app


    def train(self, app_name, budget_config=BudgetConfig()):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            train_job = self._db.create_train_job(
                budget_type=budget_config.budget_type,
                budget_amount=budget_config.budget_amount,
                app=app
            )

        # TODO: Deploy workers based on current train jobs
        
        return train_job

    def create_model(self, name, task, model_inst):
        with self._db:
            model_serialized = serialize_model(model_inst)
            model = self._db.create_model(
                name=name,
                task=task,
                model_serialized=model_serialized
            )

        return model

    def clear_all_data(self):
        with self._db:
            self._db.clear_all_data()





    