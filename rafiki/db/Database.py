import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from rafiki.constants import TrainJobStatus, \
    TrialStatus, ServiceStatus, InferenceJobStatus

from .schema import Base, TrainJob, TrainJobWorker, \
    InferenceJob, Trial, Model, User, Service, InferenceJobWorker

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

    def create_train_job(self, user_id, app, 
        app_version, task, train_dataset_uri, test_dataset_uri,
        budget_type, budget_amount):
        train_job = TrainJob(
            user_id=user_id,
            app=app,
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

    def get_train_jobs_of_app(self, app):
        train_jobs = self._session.query(TrainJob) \
            .filter(TrainJob.app == app) \
            .order_by(TrainJob.app_version.desc()).all()

        return train_jobs

    def get_train_job(self, id):
        train_job = self._session.query(TrainJob).get(id)
        return train_job

    # Returns for the latest app version unless specified
    def get_train_job_by_app_version(self, app, app_version=-1):
        query = self._session.query(TrainJob) \
            .filter(TrainJob.app == app)

        if app_version == -1:
            query = query.order_by(TrainJob.app_version.desc())
        else:
            query = query.filter(TrainJob.app_version == app_version)

        return query.first()

    def mark_train_job_as_running(self, train_job):
        train_job.status = TrainJobStatus.RUNNING
        self._session.add(train_job)
        return train_job

    def mark_train_job_as_complete(self, train_job):
        train_job.status = TrainJobStatus.COMPLETED
        train_job.datetime_completed = datetime.datetime.utcnow()
        self._session.add(train_job)
        return train_job

    ####################################
    # Train Job Workers
    ####################################

    def create_train_job_worker(self, service_id, train_job_id, model_id):
        train_job_worker = TrainJobWorker(
            train_job_id=train_job_id,
            model_id=model_id,
            service_id=service_id
        )
        self._session.add(train_job_worker)
        return train_job_worker

    def get_train_job_worker(self, service_id):
        train_job_worker = self._session.query(TrainJobWorker).get(service_id)
        return train_job_worker

    def get_workers_of_train_job(self, train_job_id):
        workers = self._session.query(TrainJobWorker) \
            .filter(TrainJobWorker.train_job_id == train_job_id).all()
        return workers

    ####################################
    # Inference Jobs
    ####################################
    
    def create_inference_job(self, user_id, train_job_id):
        inference_job = InferenceJob(
            user_id=user_id,
            train_job_id=train_job_id
        )
        self._session.add(inference_job)
        return inference_job

    def get_inference_job(self, id):
        inference_job = self._session.query(InferenceJob).get(id)
        return inference_job

    def get_inference_job_by_train_job(self, train_job_id):
        inference_job = self._session.query(InferenceJob) \
            .filter(InferenceJob.train_job_id == train_job_id).first()
        return inference_job

    def mark_inference_job_as_running(self, inference_job, 
                                    query_service_id):
        inference_job.status = InferenceJobStatus.RUNNING
        inference_job.query_service_id = query_service_id
        self._session.add(inference_job)
        return inference_job

    def mark_inference_job_as_stopped(self, inference_job):
        inference_job.status = InferenceJobStatus.STOPPED
        inference_job.datetime_stopped = datetime.datetime.utcnow()
        self._session.add(inference_job)
        return inference_job

    def get_inference_jobs_of_app(self, app):
        inference_jobs = self._session.query(InferenceJob) \
            .join(TrainJob, InferenceJob.train_job_id == TrainJob.id) \
            .filter(TrainJob.app == app) \
            .order_by(InferenceJob.datetime_started.desc()).all()

        return inference_jobs

    ####################################
    # Inference Job Workers
    ####################################

    def create_inference_job_worker(self, service_id, inference_job_id, trial_id):
        inference_job_worker = InferenceJobWorker(
            inference_job_id=inference_job_id,
            trial_id=trial_id,
            service_id=service_id
        )
        self._session.add(inference_job_worker)
        return inference_job_worker

    def get_inference_job_worker(self, service_id):
        inference_job_worker = self._session.query(InferenceJobWorker).get(service_id)
        return inference_job_worker

    def get_workers_of_inference_job(self, inference_job_id):
        workers = self._session.query(InferenceJobWorker) \
            .filter(InferenceJobWorker.inference_job_id == inference_job_id).all()
        return workers

    ####################################
    # Services
    ####################################

    def create_service(self, service_type, container_manager_type, 
                        docker_image):
        service = Service(
            service_type=service_type,
            docker_image=docker_image,
            container_manager_type=container_manager_type
        )
        self._session.add(service)
        return service

    def mark_service_as_running(self, service, container_service_id, 
                                container_service_name, replicas, hostname,
                                port, ext_hostname, ext_port):
        service.container_service_id = container_service_id
        service.container_service_name = container_service_name
        service.replicas = replicas
        service.hostname = hostname
        service.port = port
        service.ext_hostname = ext_hostname
        service.ext_port = ext_port
        service.status = ServiceStatus.RUNNING
        self._session.add(service)

    def mark_service_as_errored(self, service):
        service.status = ServiceStatus.ERRORED
        self._session.add(service)

    def mark_service_as_stopped(self, service):
        service.status = ServiceStatus.STOPPED
        service.datetime_stopped = datetime.datetime.utcnow()
        self._session.add(service)

    def get_service(self, service_id):
        service = self._session.query(Service).get(service_id)
        return service

    def get_services(self, status=None):
        query = self._session.query(Service)

        if status is not None:
            query = query.filter(Service.status == status)

        services = query.all()

        return services

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, model_serialized, docker_image):
        model = Model(
            user_id=user_id,
            name=name,
            task=task,
            model_serialized=model_serialized,
            docker_image=docker_image
        )
        self._session.add(model)
        return model

    def get_models_of_task(self, task):
        models = self._session.query(Model) \
            .filter(Model.task == task).all()

        return models

    def get_model(self, id):
        model = self._session.query(Model).get(id)
        return model

    def get_models(self):
        return self._session.query(Model).all()

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

    def get_best_trials_of_train_job(self, train_job_id, max_count=3):
        trials = self._session.query(Trial) \
            .filter(Trial.train_job_id == train_job_id) \
            .filter(Trial.status == TrainJobStatus.COMPLETED) \
            .order_by(Trial.score.desc()) \
            .limit(max_count).all()

        return trials

    def get_trials_of_app(self, app):
        trials = self._session.query(Trial) \
            .join(TrainJob, Trial.train_job_id == TrainJob.id) \
            .filter(TrainJob.app == app) \
            .order_by(Trial.datetime_started.desc())

        return trials

    def get_trials_of_train_job(self, train_job_id):
        trials = self._session.query(Trial) \
            .join(TrainJob, Trial.train_job_id == TrainJob.id) \
            .filter(TrainJob.id == train_job_id) \
            .order_by(Trial.datetime_started.desc()).all()

        return trials

    def get_completed_trials_of_train_job(self, train_job_id):
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
        

