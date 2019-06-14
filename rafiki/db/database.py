from datetime import datetime
import os
from sqlalchemy import create_engine, distinct, or_
from sqlalchemy.orm import sessionmaker

from rafiki.constants import TrainJobStatus, UserType, \
    TrialStatus, ServiceStatus, InferenceJobStatus, ModelAccessRight

from .schema import Base, TrainJob, SubTrainJob, TrainJobWorker, \
    InferenceJob, Trial, Model, User, Service, InferenceJobWorker, \
    TrialLog, Dataset

class InvalidModelAccessRightError(Exception): pass
class DuplicateModelNameError(Exception): pass
class ModelUsedError(Exception): pass
class InvalidUserTypeError(Exception): pass

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
        self._Session = sessionmaker(bind=self._engine)
        self._session = None
        self._define_tables()

    ####################################
    # Users
    ####################################

    def create_user(self, email, password_hash, user_type):
        self._validate_user_type(user_type)
        user = User(
            email=email,
            password_hash=password_hash,
            user_type=user_type
        )
        self._session.add(user)
        return user

    def ban_user(self, user):
        user.banned_date = datetime.utcnow()
        self._session.add(user)

    def get_user_by_email(self, email):
        user = self._session.query(User).filter(User.email == email).first()
        return user

    def get_users(self):
        users = self._session.query(User).all()
        return users

    def _validate_user_type(self, user_type):
        if user_type not in [UserType.SUPERADMIN, UserType.ADMIN, UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER]:
            raise InvalidUserTypeError()

    ####################################
    # Datasets
    ####################################

    def create_dataset(self, name, task, size_bytes, store_dataset_id, owner_id):
        dataset = Dataset(
            name=name,
            task=task,
            size_bytes=size_bytes,
            store_dataset_id=store_dataset_id,
            owner_id=owner_id
        )
        self._session.add(dataset)
        return dataset

    def get_dataset(self, id):
        dataset = self._session.query(Dataset).get(id)
        return dataset

    def get_datasets(self, user_id, task=None):
        query = self._session.query(Dataset) \
            .filter(Dataset.owner_id == user_id)

        if task is not None:
            query = query.filter(Dataset.task == task)
        
        datasets = query.all()
        return datasets
    
    ####################################
    # Train Jobs
    ####################################

    def create_train_job(self, user_id, app, app_version, task, budget,
                        train_dataset_id, val_dataset_id):

        train_job = TrainJob(
            user_id=user_id,
            app=app,
            app_version=app_version,
            task=task,
            budget=budget,
            train_dataset_id=train_dataset_id,
            val_dataset_id=val_dataset_id
        )
        self._session.add(train_job)
        return train_job

    def get_train_jobs_of_app(self, app):
        train_jobs = self._session.query(TrainJob) \
            .filter(TrainJob.app == app) \
            .order_by(TrainJob.app_version.desc()).all()

        return train_jobs

    def get_train_jobs_by_user(self, user_id):
        train_jobs = self._session.query(TrainJob) \
            .filter(TrainJob.user_id == user_id).all()

        return train_jobs

    def get_train_job(self, id):
        train_job = self._session.query(TrainJob).get(id)
        return train_job

    def get_train_jobs_by_status(self, status):
        job_ids = self._session.query(distinct(SubTrainJob.train_job_id)) \
            .filter(SubTrainJob.status == status).all()
        return self._session.query(TrainJob) \
            .filter(TrainJob.id.in_(job_ids)).all()

    # Returns for the latest app version unless specified
    def get_train_job_by_app_version(self, app, app_version=-1):
        # pylint: disable=E1111
        query = self._session.query(TrainJob) \
            .filter(TrainJob.app == app)

        if app_version == -1:
            query = query.order_by(TrainJob.app_version.desc())
        else:
            query = query.filter(TrainJob.app_version == app_version)

        return query.first()

    ####################################
    # Sub Train Jobs
    ####################################  

    def create_sub_train_job(self, train_job_id, model_id, user_id):
        sub_train_job = SubTrainJob(
            train_job_id=train_job_id,
            model_id=model_id,
            user_id=user_id
        )
        self._session.add(sub_train_job)
        return sub_train_job

    def get_sub_train_jobs_of_train_job(self, train_job_id):
        sub_train_jobs = self._session.query(SubTrainJob) \
            .filter(SubTrainJob.train_job_id == train_job_id) \
            .all()

        return sub_train_jobs 

    def get_sub_train_job(self, id):
        sub_train_job = self._session.query(SubTrainJob).get(id)
        return sub_train_job

    def mark_sub_train_job_as_running(self, sub_train_job):
        sub_train_job.status = TrainJobStatus.RUNNING
        sub_train_job.datetime_stopped = None
        self._session.add(sub_train_job)
        return sub_train_job

    def mark_sub_train_job_as_stopped(self, sub_train_job):
        sub_train_job.status = TrainJobStatus.STOPPED
        sub_train_job.datetime_stopped = datetime.utcnow()
        self._session.add(sub_train_job)
        return sub_train_job

    ####################################
    # Train Job Workers
    ####################################

    def create_train_job_worker(self, service_id, train_job_id, sub_train_job_id):
        train_job_worker = TrainJobWorker(
            train_job_id=train_job_id,
            sub_train_job_id=sub_train_job_id,
            service_id=service_id
        )
        self._session.add(train_job_worker)
        return train_job_worker

    def get_train_job_worker(self, service_id):
        train_job_worker = self._session.query(TrainJobWorker).get(service_id)
        return train_job_worker

    def get_workers_of_sub_train_job(self, sub_train_job_id):
        workers = self._session.query(TrainJobWorker) \
            .filter(TrainJobWorker.sub_train_job_id == sub_train_job_id).all()
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

    def get_inference_job_by_predictor(self, predictor_service_id):
        inference_job = self._session.query(InferenceJob) \
            .filter(InferenceJob.predictor_service_id == predictor_service_id).first()
        return inference_job

    def get_running_inference_job_by_train_job(self, train_job_id):
        inference_job = self._session.query(InferenceJob) \
            .filter(InferenceJob.train_job_id == train_job_id) \
            .filter(InferenceJob.status == InferenceJobStatus.RUNNING).first()
        return inference_job

    def get_inference_jobs_by_user(self, user_id):
        inference_jobs = self._session.query(InferenceJob) \
            .filter(InferenceJob.user_id == user_id).all()

        return inference_jobs

    def update_inference_job(self, inference_job, predictor_service_id):
        inference_job.predictor_service_id = predictor_service_id
        self._session.add(inference_job)
        return inference_job
    
    def mark_inference_job_as_running(self, inference_job):
        inference_job.status = InferenceJobStatus.RUNNING
        inference_job.datetime_stopped = None
        return inference_job

    def mark_inference_job_as_stopped(self, inference_job):
        inference_job.status = InferenceJobStatus.STOPPED
        inference_job.datetime_stopped = datetime.utcnow()
        self._session.add(inference_job)
        return inference_job

    def mark_inference_job_as_errored(self, inference_job):
        inference_job.status = InferenceJobStatus.ERRORED
        inference_job.datetime_stopped = datetime.utcnow()
        self._session.add(inference_job)
        return inference_job

    def get_inference_jobs_of_app(self, app):
        inference_jobs = self._session.query(InferenceJob) \
            .join(TrainJob, InferenceJob.train_job_id == TrainJob.id) \
            .filter(TrainJob.app == app) \
            .order_by(InferenceJob.datetime_started.desc()).all()

        return inference_jobs
    
    def get_inference_jobs_by_status(self, status):
        jobs = self._session.query(InferenceJob) \
            .filter(InferenceJob.status == status).all()

        return jobs

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
                        docker_image, requirements):
        service = Service(
            service_type=service_type,
            docker_image=docker_image,
            container_manager_type=container_manager_type,
            requirements=requirements
        )
        self._session.add(service)
        return service

    def mark_service_as_deploying(self, service, container_service_id, 
                                container_service_name, replicas, hostname,
                                port, ext_hostname, ext_port):
        service.container_service_id = container_service_id
        service.container_service_name = container_service_name
        service.replicas = replicas
        service.hostname = hostname
        service.port = port
        service.ext_hostname = ext_hostname
        service.ext_port = ext_port
        service.status = ServiceStatus.DEPLOYING
        self._session.add(service)

    def mark_service_as_running(self, service):
        service.status = ServiceStatus.RUNNING
        service.datetime_stopped = None
        self._session.add(service)

    def mark_service_as_errored(self, service):
        service.status = ServiceStatus.ERRORED
        service.datetime_stopped = datetime.utcnow()
        self._session.add(service)

    def mark_service_as_stopped(self, service):
        service.status = ServiceStatus.STOPPED
        service.datetime_stopped = datetime.utcnow()
        self._session.add(service)

    def get_service(self, service_id):
        service = self._session.query(Service).get(service_id)
        return service

    def get_services(self, status=None):
        query = self._session.query(Service)

        if status is not None:
            # pylint: disable=E1111
            query = query.filter(Service.status == status)

        services = query.all()

        return services

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, model_file_bytes, 
                    model_class, docker_image, dependencies, access_right):
        
        self._validate_model_access_right(access_right)

        model = Model(
            user_id=user_id,
            name=name,
            task=task,
            model_file_bytes=model_file_bytes,
            model_class=model_class,
            docker_image=docker_image,
            dependencies=dependencies,
            access_right=access_right
        )
        self._session.add(model)
        return model

    def delete_model(self, model):
        # Ensure that model has no referencing jobs
        sub_train_job = self._session.query(SubTrainJob) \
            .filter(SubTrainJob.model_id == model.id).first()

        if sub_train_job is not None:
            raise ModelUsedError()

        self._session.delete(model)

    def get_available_models(self, user_id, task=None):
        # Get public models or user's own models
        query = self._session.query(Model) \
            .filter(or_(Model.access_right == ModelAccessRight.PUBLIC, Model.user_id == user_id))
        if task is not None:
            query = query.filter(Model.task == task)
        models = query.all()
        return models

    def get_model_by_name(self, user_id, name):
        model = self._session.query(Model) \
            .filter(Model.user_id == user_id) \
            .filter(Model.name == name).first()
        
        return model

    def get_model(self, id):
        model = self._session.query(Model).get(id)
        return model

    def _validate_model_access_right(self, access_right):
        if access_right not in [ModelAccessRight.PUBLIC, ModelAccessRight.PRIVATE]:
            raise InvalidModelAccessRightError()

    ####################################
    # Trials
    ####################################

    def create_trial(self, sub_train_job_id, model_id):
        trial = Trial(
            sub_train_job_id=sub_train_job_id,
            model_id=model_id
        )
        self._session.add(trial)
        return trial

    def get_trial(self, id):
        trial = self._session.query(Trial) \
            .join(SubTrainJob, Trial.sub_train_job_id == SubTrainJob.id) \
            .filter(Trial.id == id) \
            .first()

        return trial

    def get_trial_logs(self, id):
        trial_logs = self._session.query(TrialLog) \
            .filter(TrialLog.trial_id == id) \
            .all()
            
        return trial_logs
    
    def get_best_trials_of_train_job(self, train_job_id, max_count=2):
        trials = self._session.query(Trial) \
            .join(SubTrainJob, Trial.sub_train_job_id == SubTrainJob.id) \
            .filter(SubTrainJob.train_job_id == train_job_id) \
            .filter(Trial.status == TrialStatus.COMPLETED) \
            .order_by(Trial.score.desc()) \
            .limit(max_count).all()

        return trials

    def get_trials_of_sub_train_job(self, sub_train_job_id):
        trials = self._session.query(Trial) \
            .filter(Trial.sub_train_job_id == sub_train_job_id) \
            .order_by(Trial.datetime_started.desc()).all()

        return trials

    def get_trials_of_app(self, app):
        trials = self._session.query(Trial) \
            .join(SubTrainJob, Trial.sub_train_job_id == SubTrainJob.id) \
            .join(TrainJob, TrainJob.id == SubTrainJob.train_job_id) \
            .filter(TrainJob.app == app) \
            .order_by(Trial.datetime_started.desc())

        return trials

    def mark_trial_as_running(self, trial, knobs):
        trial.status = TrialStatus.RUNNING
        trial.knobs = knobs
        self._session.add(trial)
        return trial

    def mark_trial_as_errored(self, trial):
        trial.status = TrialStatus.ERRORED
        trial.datetime_stopped = datetime.utcnow()
        self._session.add(trial)
        return trial

    def mark_trial_as_complete(self, trial, score, params_file_path):
        trial.status = TrialStatus.COMPLETED
        trial.score = score
        trial.datetime_stopped = datetime.utcnow()
        trial.params_file_path = params_file_path
        self._session.add(trial)
        return trial

    def add_trial_log(self, trial, line, level):
        trial_log = TrialLog(trial_id=trial.id, line=line, level=level)
        self._session.add(trial_log)
        return trial_log

    def mark_trial_as_terminated(self, trial):
        trial.status = TrialStatus.TERMINATED
        trial.datetime_stopped = datetime.utcnow()
        self._session.add(trial)
        return trial

    ####################################
    # Others
    ####################################
    
    def __enter__(self):
        self.connect()

    def connect(self):
        self._session = self._Session()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def commit(self):
        try:
            self._session.commit()
        except Exception as e:
            # Check if error is due to duplicate model name
            if 'model_name_user_id_key' in str(e):
                self._session.rollback()
                raise DuplicateModelNameError()
            else:
                raise e

    # Ensures that future database queries load fresh data from underlying database
    def expire(self):
        self._session.expire_all()

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
