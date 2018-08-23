

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
            self._db.commit()
            app = self._db.create_app(name, task, train_dataset.id, test_dataset.id)


    def get_app_status(self, name):
        with self._db:
            app = self._db.get_app_by_name(name)
            train_dataset = self._db.get_dataset(app.train_dataset_id)
            test_dataset = self._db.get_dataset(app.test_dataset_id)
            train_jobs = self._db.get_train_jobs_by_app(app.id)
            best_trials = self._db.get_best_trials_by_app(app.id, max_count=3)
            best_trials_models = [self._db.get_model(x.model_id) for x in best_trials]

            return {
                'name': app.name,
                'task': app.task,
                'datetime_created': app.datetime_created,
                'train_dataset': {
                    'dataset_type': train_dataset.dataset_type,
                    'params': train_dataset.params,
                },
                'test_dataset': {
                    'dataset_type': test_dataset.dataset_type,
                    'params': test_dataset.params,
                },
                'train_jobs': [
                    {
                        'status': train_job.status,
                        'datetime_started': train_job.datetime_started,
                        'datetime_completed': train_job.datetime_completed,
                        'budget_type': train_job.budget_type,
                        'budget_amount': train_job.budget_amount
                    }
                        for train_job in train_jobs
                ],
                'best_trials': [
                    {
                        'hyperparameters': trial.hyperparameters,
                        'datetime_started': trial.datetime_started,
                        'model_name': model.name,
                        'score': trial.score
                    }
                        for (trial, model) in zip(best_trials, best_trials_models)
                ]
            }

    def train(self, app_name, budget_config=BudgetConfig()):
        with self._db:
            app = self._db.get_app_by_name(app_name)
            train_job = self._db.create_train_job(
                budget_type=budget_config.budget_type,
                budget_amount=budget_config.budget_amount,
                app_id=app.id
            )

        # TODO: Deploy workers based on current train jobs

    # def predict(self, app_name, queries):
    #     with self._db:
            
    #     model_inst = unserialize_model(model.model_serialized)
    #     model_inst.init(hyperparameters)

    def create_model(self, name, task, model_inst):
        with self._db:
            model_serialized = serialize_model(model_inst)
            model = self._db.create_model(
                name=name,
                task=task,
                model_serialized=model_serialized
            )


    def clear_all_data(self):
        with self._db:
            self._db.clear_all_data()





    