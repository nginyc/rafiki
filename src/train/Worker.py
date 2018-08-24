import time
import logging
import random
import dill
import traceback

from common import unserialize_model, create_tuner, \
    propose_with_tuner, train_tuner, BudgetType
from db import DatabaseConfig, Database, TrainJobStatus, TrialStatus

logger = logging.getLogger(__name__)

class NoModelsForTaskException(Exception):
    pass

class InvalidBudgetTypeException(Exception):
    pass

class Worker(object):
    TRAIN_JOB_POLL_INTERVAL_SECONDS = 10

    def __init__(self, database_config=DatabaseConfig()):
        self._db = Database(database_config)

    def start(self):
        with self._db:
            while True:
                has_train_job = self._do_train_loop()
                
                if not has_train_job:
                    sleep_secs = self.TRAIN_JOB_POLL_INTERVAL_SECONDS
                    logger.info('No uncompleted train jobs left. Sleeping for {}s...' \
                        .format(sleep_secs))
                    time.sleep(sleep_secs)

                
    # Returns False if there is no train job
    def _do_train_loop(self):
        train_job = self._select_train_job()

        if train_job is None:
            return False

        logger.info(
            'Working on train job of ID {}...' \
                .format(train_job.id)
        )

        try:
            app = self._db.get_app(train_job.app_id)
            models = self._db.get_models_by_task(app.task)
            train_dataset = self._db.get_dataset(app.train_dataset_id)
            test_dataset = self._db.get_dataset(app.test_dataset_id)
            self._do_trial(models, train_job, train_dataset, test_dataset)
            self._check_train_job_budget(train_job)
        except Exception as error:
            logger.error('Error while running train job:')
            logger.error(traceback.format_exc())

        return True

    def _do_trial(self, models, train_job, train_dataset, test_dataset):
        
        (model, hyperparameters) = \
            self._do_hyperparameter_selection(models, train_job)
                
        trial = self._db.create_trial(
            model=model, 
            train_job_id=train_job.id, 
            hyperparameters=hyperparameters
        )
        self._db.commit()
        
        try:
            model_inst = unserialize_model(model.model_serialized)
            model_inst.init(hyperparameters)

            # Train model
            model_inst.train(train_dataset.config)

            # Evaluate model
            score = model_inst.evaluate(test_dataset.config)
            
            parameters = model_inst.dump_parameters()
            model_inst.destroy()

            self._db.mark_trial_as_complete(
                trial,
                score=score,
                parameters=parameters
            )
            self._db.commit()

        except Exception as error:
            logger.error('Error while running trial:')
            logger.error(traceback.format_exc())

            self._db.mark_trial_as_errored(trial)
            self._db.commit()

        return True

    # Updates train job based on budget
    def _check_train_job_budget(self, train_job):
        if train_job.budget_type == BudgetType.TRIAL_COUNT:
            trials = self._db.get_completed_trials_by_train_job(train_job.id)
            max_trials = train_job.budget_amount 
            if len(trials) >= max_trials:
                logger.info('Train job has reached target trial count')
                self._db.mark_train_job_as_complete(train_job)
                self._db.commit()

        else:
            raise InvalidBudgetTypeException()


    # Returns an uncompleted train job
    def _select_train_job(self):
        train_jobs = self._db.get_uncompleted_train_jobs()

        if len(train_jobs) == 0:
            return None

        # TODO: Better train job selection
        return train_jobs[0]

    # Returns a set of hyperparameter values
    def _do_hyperparameter_selection(self, models, train_job):
        if len(models) == 0:
            raise NoModelsForTaskException()
        
        # TODO: Better hyperparameter values selection

        # Randomly pick a model
        model = random.choice(models)

        # Pick hyperparameter values
        tuner = self._get_tuner_for_model(train_job, model)
        hyperparameters = propose_with_tuner(tuner)

        return (model, hyperparameters)
        
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
