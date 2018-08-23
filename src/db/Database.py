import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .Base import Base
from .Dataset import Dataset
from .TrainJob import TrainJob, TrainJobStatus
from .Trial import Trial, TrialStatus
from .Model import Model
from .App import App

class Database(object):
    def __init__(self, database_config):
        db_connection_url = self._make_connection_url(database_config)
        self._engine = create_engine(db_connection_url, echo=True)
        self._session = None
        self._define_tables()
        
    def __enter__(self):
        self.connect()        

    def connect(self):
        Session = sessionmaker(bind=self._engine)
        self._session = Session()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def commit(self):
        self._session.commit()

    def disconnect(self):
        if self._session is not None:
            self._session.commit()
            self._session.close()
            self._session = None

    def create_app(self, name, task, train_dataset, test_dataset):
        app = App(
            name=name, 
            task=task,
            train_dataset=train_dataset,
            test_dataset=test_dataset
        )
        self._session.add(app)
        return app

    def get_app_by_name(self, name):
        app = self._session.query(App).filter(App.name == name).first()
        return app

    def get_app(self, id):
        app = self._session.query(App).get(id)
        return app

    def get_dataset(self, id):
        dataset = self._session.query(Dataset).get(id)
        return dataset

    def create_dataset(self, dataset_type, params):
        dataset = Dataset(dataset_type=dataset_type, params=params)
        self._session.add(dataset)
        return dataset

    def create_train_job(self, budget_type, budget_amount, app):
        train_job = TrainJob(
            budget_type=budget_type, 
            budget_amount=budget_amount,
            app=app
        )
        self._session.add(train_job)
        return train_job

    def get_uncompleted_train_jobs(self):
        train_jobs = self._session.query(TrainJob) \
            .filter(TrainJob.status == TrainJobStatus.STARTED).all()

        return train_jobs

    def mark_train_job_as_complete(self, train_job):
        train_job.status = TrainJobStatus.COMPLETE
        train_job.datetime_completed = datetime.datetime.utcnow()
        self._session.add(train_job)
        return train_job

    def create_model(self, name, task, model_serialized):
        model = Model(
            name=name,
            task=task,
            model_serialized=model_serialized
        )
        self._session.add(model)
        return model

    def get_models_by_task(self, task):
        models = self._session.query(Model) \
            .filter(Model.task == task).all()

        return models

    def get_completed_trials_by_train_job(self, train_job):
        trials = self._session.query(Trial) \
            .filter(Trial.status == TrainJobStatus.COMPLETE) \
            .filter(Trial.train_job == train_job).all()

        return trials

    def create_trial(self, model, train_job, 
                    hyperparameters):
        trial = Trial(
            model=model,
            train_job=train_job,
            hyperparameters=hyperparameters
        )
        self._session.add(trial)
        return trial

    def mark_trial_as_errored(self, trial):
        trial.status = TrialStatus.ERRORED
        self._session.add(trial)
        return trial

    def mark_trial_as_complete(self, trial, score, parameters):
        trial.status = TrialStatus.COMPLETE
        trial.score = score
        trial.datetime_completed = datetime.datetime.utcnow()
        trial.parameters = parameters
        self._session.add(trial)
        return trial

    def clear_all_data(self):
        for table in reversed(Base.metadata.sorted_tables):
            self._session.execute(table.delete())

    def _make_connection_url(self, database_config):
        return 'postgresql://{}:{}@{}:{}/{}'.format(
            database_config.user,
            database_config.password,
            database_config.host,
            database_config.port,
            database_config.db
        )

    def _define_tables(self):
        Base.metadata.create_all(bind=self._engine)
        

