import time
import logging
import random
import dill
import traceback

from common import TrainJobStatus, TrialStatus
from model import unserialize_model
from db import Database

from .tuner import propose_with_tuner, train_tuner, create_tuner

logger = logging.getLogger(__name__)

class NoSuchTrainJobException(Exception):
    pass

class NoSuchModelException(Exception):
    pass

class Worker(object):
    def __init__(self, worker_id, db=Database()):
        self._db = db
        self._worker_id = worker_id

    def start(self):
        logger.info(
            'Starting worker of ID {}...'.format(self._worker_id)
        )

        with self._db:
            worker = self._db.get_train_job_worker(self._worker_id)
            self._db.mark_train_job_worker_as_running(worker)
        
        while True:
            try:
                model_id = None
                train_job_id = None

                with self._db:
                    worker = self._db.get_train_job_worker(self._worker_id)
                    model_id = worker.model_id
                    train_job_id = worker.train_job_id
                        
                self._do_new_trial(train_job_id, model_id)
                
            except Exception:
                with self._db:
                    worker = self._db.get_train_job_worker(self._worker_id)
                    self._db.mark_train_job_worker_as_errored(worker)

                logger.error('Error while running worker:')
                logger.error(traceback.format_exc())
                logger.error('Exiting worker...')
                exit(1)

    def _do_new_trial(self, train_job_id, model_id):

        self._db.connect()
        (train_dataset_uri, test_dataset_uri,
            model_serialized, hyperparameters, trial_id) = \
                self._create_new_trial(train_job_id, model_id)
        self._db.disconnect()

        try:
            model_inst = unserialize_model(model_serialized)
            model_inst.init(hyperparameters)

            # Train model
            model_inst.train(train_dataset_uri)

            # Evaluate model
            score = model_inst.evaluate(test_dataset_uri)
            
            parameters = model_inst.dump_parameters()
            model_inst.destroy()

            with self._db:
                trial = self._db.get_trial(trial_id)
                self._db.mark_trial_as_complete(
                    trial,
                    score=score,
                    parameters=parameters
                )

        except Exception as error:
            logger.error('Error while running trial:')
            logger.error(traceback.format_exc())

            with self._db:
                trial = self._db.get_trial(trial_id)
                self._db.mark_trial_as_errored(trial)

    def _create_new_trial(self, train_job_id, model_id):
        train_job = self._db.get_train_job(train_job_id)
        if train_job is None:
            raise NoSuchTrainJobException('ID: {}'.format(train_job_id))

        model = self._db.get_model(model_id)
        if model is None:
            raise NoSuchModelException('ID: {}'.format(model_id))
    
        hyperparameters = self._do_hyperparameter_selection(train_job, model)

        trial = self._db.create_trial(
            model=model, 
            train_job_id=train_job.id, 
            hyperparameters=hyperparameters
        )
        self._db.commit()

        return (
            train_job.train_dataset_uri,
            train_job.test_dataset_uri,
            model.model_serialized,
            hyperparameters,
            trial.id
        )

    # Returns a set of hyperparameter values
    def _do_hyperparameter_selection(self, train_job, model):
        # Pick hyperparameter values
        tuner = self._get_tuner_for_model(train_job, model)
        hyperparameters = propose_with_tuner(tuner)

        return hyperparameters
        
    # Retrieves/creates a tuner for the model for the associated train job
    def _get_tuner_for_model(self, train_job, model):
        # Instantiate tuner
        model_inst = unserialize_model(model.model_serialized)
        hyperparameters_config = model_inst.get_hyperparameter_config()
        tuner = create_tuner(hyperparameters_config)

        # Train tuner
        trials = self._db.get_completed_trials_by_train_job(train_job.id)
        model_trial_history = [(x.hyperparameters, x.score) for x in trials if x.model_id == model.id]
        (hyperparameters_list, scores) = [list(x) for x in zip(*model_trial_history)] \
            if len(model_trial_history) > 0 else ([], [])
        tuner = train_tuner(tuner, hyperparameters_list, scores)

        return tuner
