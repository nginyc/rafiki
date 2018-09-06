import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from common import TrainJobStatus, TrialStatus, TrainJobWorkerStatus

from .schema import Base, TrainJob, TrainJobWorker, \
    InferenceJob, Trial, Model, User

class Database(object):
    def __init__(self, 
        host=os.environ.get('POSTGRES_HOST', 'localhost'), 
        port=os.environ.get('POSTGRES_PORT', 5432),
        user=os.environ.get('POSTGRES_USER', 'rafiki'),
        db=os.environ.get('POSTGRES_DB', 'rafiki'),
        password=os.environ.get('POSTGRES_PASSWORD', 'rafiki')):

        db_connection_url = self._make_connection_url(
            host=host, 
            port=port, 
            db=db,
            user=user, 
            password=password
        )
        
        self._engine = create_engine(db_connection_url)
        self._session = None
        self._define_tables()

    ####################################
    # Users
    ####################################

    def create_user(self, email, password_hash, user_type):
        user = User(
            email=email,
            password_hash=password_hash,
            user_type=user_type
        )
        self._session.add(user)
        return user

    def get_user_by_email(self, email):
        user = self._session.query(User).filter(User.email == email).first()
        return user

    ####################################
    # Train Jobs
    ####################################

    def create_train_job(self, user_id, app_name, 
        app_version, task, train_dataset_uri, test_dataset_uri,
        budget_type, budget_amount):
        train_job = TrainJob(
            user_id=user_id,
            app_name=app_name,
            app_version=app_version,
            task=task,
            train_dataset_uri=train_dataset_uri,
            test_dataset_uri=test_dataset_uri,
            budget_type=budget_type, 
            budget_amount=budget_amount
        )
        self._session.add(train_job)
        return train_job

    def get_uncompleted_train_jobs(self):
        train_jobs = self._session.query(TrainJob) \
            .filter(TrainJob.status == TrainJobStatus.STARTED).all()

        return train_jobs

    def get_train_jobs_by_app(self, app_name):
        train_jobs = self._session.query(TrainJob) \
            .filter(TrainJob.app_name == app_name) \
            .order_by(TrainJob.datetime_started.desc()).all()

        return train_jobs

    def get_train_job(self, id):
        train_job = self._session.query(TrainJob).get(id)
        return train_job

    def mark_train_job_as_complete(self, train_job):
        train_job.status = TrainJobStatus.COMPLETED
        train_job.datetime_completed = datetime.datetime.utcnow()
        self._session.add(train_job)
        return train_job

    ####################################
    # Inference Jobs
    ####################################
    
    def create_inference_job(self, user_id, app_name):
        inference_job = InferenceJob(
            user_id=user_id,
            app_name=app_name
        )
        self._session.add(inference_job)
        return inference_job

    def get_inference_jobs_by_app(self, app_name):
        inference_jobs = self._session.query(InferenceJob) \
            .filter(InferenceJob.app_name == app_name) \
            .order_by(InferenceJob.datetime_started.desc()).all()

        return inference_jobs

    ####################################
    # Train Job Workers
    ####################################

    def create_train_job_worker(self, train_job_id, model_id):
        worker = TrainJobWorker(
            train_job_id=train_job_id,
            model_id=model_id
        )
        self._session.add(worker)
        return worker

    def update_train_job_worker(self, worker, service_id=None, replicas=None):
        if service_id is not None:
            worker.service_id = service_id

        if replicas is not None:
            worker.replicas = replicas

        self._session.add(worker)
        return worker

    def mark_train_job_worker_as_running(self, worker):
        worker.status = TrainJobWorkerStatus.RUNNING
        self._session.add(worker)
        return worker

    def mark_train_job_worker_as_stopped(self, worker):
        worker.status = TrainJobWorkerStatus.STOPPED
        self._session.add(worker)
        return worker

    def get_train_job_workers_by_train_job(self, train_job_id):
        workers = self._session.query(TrainJobWorker) \
            .filter(TrainJobWorker.train_job_id == train_job_id).all()
        return workers

    def get_train_job_worker(self, id):
        worker = self._session.query(TrainJobWorker).get(id)
        return worker

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, model_serialized, docker_image_name):
        model = Model(
            user_id=user_id,
            name=name,
            task=task,
            model_serialized=model_serialized,
            docker_image_name=docker_image_name
        )
        self._session.add(model)
        return model

    def get_models_by_task(self, task):
        models = self._session.query(Model) \
            .filter(Model.task == task).all()

        return models

    def get_model(self, id):
        model = self._session.query(Model).get(id)
        return model
    
    def get_models(self):
        models = self._session.query(Model).all()
        return models

    ####################################
    # Trials
    ####################################

    def create_trial(self, model_id, train_job_id, 
                    hyperparameters):
        trial = Trial(
            model_id=model_id,
            train_job_id=train_job_id,
            hyperparameters=hyperparameters
        )
        self._session.add(trial)
        return trial

    def get_trial(self, id):
        trial = self._session.query(Trial) \
            .join(TrainJob, Trial.train_job_id == TrainJob.id) \
            .filter(Trial.id == id) \
            .first()

        return trial

    def get_best_trials_by_app(self, app_name, max_count=3):
        trials = self._session.query(Trial) \
            .join(TrainJob, Trial.train_job_id == TrainJob.id) \
            .filter(TrainJob.app_name == app_name) \
            .filter(Trial.status == TrainJobStatus.COMPLETED) \
            .order_by(Trial.score.desc()) \
            .limit(max_count).all()

        return trials

    def get_trials_by_app(self, app_name):
        trials = self._session.query(Trial) \
            .join(TrainJob, Trial.train_job_id == TrainJob.id) \
            .filter(TrainJob.app_name == app_name) \
            .order_by(Trial.datetime_started.desc())

        return trials

    def get_trials_by_train_job(self, train_job_id):
        trials = self._session.query(Trial) \
            .join(TrainJob, Trial.train_job_id == TrainJob.id) \
            .filter(TrainJob.id == train_job_id) \
            .order_by(Trial.datetime_started.desc()).all()

        return trials

    def get_completed_trials_by_train_job(self, train_job_id):
        trials = self._session.query(Trial) \
            .filter(Trial.status == TrainJobStatus.COMPLETED) \
            .filter(Trial.train_job_id == train_job_id).all()

        return trials

    def mark_trial_as_errored(self, trial):
        trial.status = TrialStatus.ERRORED
        self._session.add(trial)
        return trial

    def mark_trial_as_complete(self, trial, score, parameters):
        trial.status = TrialStatus.COMPLETED
        trial.score = score
        trial.datetime_completed = datetime.datetime.utcnow()
        trial.parameters = parameters
        self._session.add(trial)
        return trial

    ####################################
    # Others
    ####################################
    
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
            
    def clear_all_data(self):
        for table in reversed(Base.metadata.sorted_tables):
            self._session.execute(table.delete())

    def _make_connection_url(self, host, port, db, user, password):
        return 'postgresql://{}:{}@{}:{}/{}'.format(
            user, password, host, port, db
        )

    def _define_tables(self):
        Base.metadata.create_all(bind=self._engine)
        

