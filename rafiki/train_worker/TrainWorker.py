import time
import logging
import os
import traceback
import pprint

from rafiki.constants import TrainJobStatus, TrialStatus, BudgetType
from rafiki.utils.model import unserialize_model
from rafiki.db import Database
from rafiki.client import Client

logger = logging.getLogger(__name__)

class InvalidTrainJobException(Exception):
    pass

class InvalidModelException(Exception):
    pass

class InvalidBudgetTypeException(Exception):
    pass

class InvalidWorkerException(Exception):
    pass

class TrainWorker(object):
    def __init__(self, service_id, db=Database()):
        self._service_id = service_id
        self._db = db
        self._trial_id = None
        self._client = self._make_client()

    def start(self):
        logger.info('Starting train worker for service of ID "{}"...' \
            .format(self._service_id))
            
        advisor_id = None
        while True:
            self._db.connect()
            (budget_type, budget_amount, model_id,
            model_serialized, train_job_id, 
            train_dataset_uri, test_dataset_uri) = self._read_worker_info()

            if self._if_budget_reached(budget_type, budget_amount, train_job_id, model_id):
                # If budget reached
                logger.info('Budget for train job has reached')

                self._stop_worker()
                if advisor_id is not None:
                    self._delete_advisor(advisor_id)

                break

            # If not created, create a Rafiki advisor for train worker to propose knobs in trials
            if advisor_id is None:
                logger.info('Creating Rafiki advisor...')
                try: 
                    advisor_id = self._create_advisor(model_serialized)
                    logger.info('Created advisor of ID "{}"'.format(advisor_id))
                except Exception as e:
                    # Throw just a warning - likely that another worker has stopped the service
                    logger.error('Error while creating advisor for worker:')
                    logger.error(traceback.format_exc())
                    raise e

            # Create a new trial
            logger.info('Starting trial...')
            logger.info('Requesting for knobs proposal from advisor...')
            (proposal_id, knobs) = self._get_proposal_from_advisor(advisor_id)
            logger.info('Received proposal of ID "{}" from advisor:'.format(proposal_id))
            logger.info(pprint.pformat(knobs))
            logger.info('Creating new trial in DB...')
            trial = self._create_new_trial(model_id, train_job_id, knobs)
            self._trial_id = trial.id
            logger.info('Created trial of ID "{}" in DB'.format(trial.id))

            # Don't keep DB connection while training model
            self._db.disconnect()

            # Perform trial & record results
            score = 0
            try:
                logger.info('Starting trial...')
                logger.info('Training & evaluating model...')
                self._db.mark_trial_as_errored
                (score, parameters) = self._train_and_evaluate_model(model_serialized, 
                                                                    knobs,
                                                                    train_dataset_uri, 
                                                                    test_dataset_uri)
                logger.info('Trial score: {}'.format(score))
                with self._db:
                    logger.info('Marking trial as complete in DB...')
                    trial = self._db.get_trial(self._trial_id)
                    self._db.mark_trial_as_complete(
                        trial,
                        score=score,
                        parameters=parameters
                    )
                    self._trial_id = None
            except Exception:
                logger.error('Error while running trial:')
                logger.error(traceback.format_exc())

                logger.info('Marking trial as errored in DB...')
                with self._db:
                    trial = self._db.get_trial(self._trial_id)
                    self._db.mark_trial_as_errored(trial)
                    self._trial_id = None

            # Report results of trial to advisor
            try:
                logger.info('Sending result of proposal to advisor...')
                self._set_result_of_proposal(advisor_id, proposal_id, score)
            except Exception:
                logger.error('Error while sending result of proposal to advisor:')
                logger.error(traceback.format_exc())
            
    def stop(self):
        # If worker is currently running a trial, mark it has terminated
        logger.info('Marking trial as terminated in DB...')
        try:
            if self._trial_id is not None: 
                with self._db:
                    trial = self._db.get_trial(self._trial_id)
                    self._db.mark_trial_as_terminated(trial)

        except Exception:
            logger.error('Error marking trial as terminated:')
            logger.error(traceback.format_exc())

    def _train_and_evaluate_model(self, model_serialized, knobs, train_dataset_uri, 
                                test_dataset_uri):
        model_inst = unserialize_model(model_serialized)
        model_inst.init(knobs)

        # Train model
        model_inst.train(train_dataset_uri)

        # Evaluate model
        score = model_inst.evaluate(test_dataset_uri)
        parameters = model_inst.dump_parameters()
        model_inst.destroy()

        return (score, parameters)

    # Creates a new trial in the DB
    def _create_new_trial(self, model_id, train_job_id, knobs):
        trial = self._db.create_trial(
            model_id=model_id, 
            train_job_id=train_job_id, 
            knobs=knobs
        )
        self._db.commit()
        return trial

    # Returns a set of knob values
    def _get_proposal_from_advisor(self, advisor_id):
        res = self._client.generate_proposal(advisor_id)
        proposal_id = res['id']
        knobs = res['knobs']
        return (proposal_id, knobs)

    # Send result of proposal to advisor
    def _set_result_of_proposal(self, advisor_id, proposal_id, score):
        self._client.set_result_of_proposal(advisor_id, proposal_id, score)

    def _stop_worker(self):
        try:
            self._client.stop_train_job_worker(self._service_id)
        except Exception:
            # Throw just a warning - likely that another worker has stopped the service
            logger.warning('Error while stopping train job worker service:')
            logger.warning(traceback.format_exc())
        
    def _create_advisor(self, model_serialized):
        # Retrieve knob config for model of worker 
        model_inst = unserialize_model(model_serialized)
        knob_config = model_inst.get_knob_config()

        # Create advisor associated with worker
        res = self._client.create_advisor(knob_config, advisor_id=self._service_id)
        advisor_id = res['id']
        return advisor_id

    # Delete advisor
    def _delete_advisor(self, advisor_id):
        try:
            self._client.delete_advisor(advisor_id)
        except Exception:
            # Throw just a warning - not critical for advisor to be deleted
            logger.warning('Error while deleting advisor:')
            logger.warning(traceback.format_exc())

    # Returns whether the worker reached its budget
    def _if_budget_reached(self, budget_type, budget_amount, train_job_id, model_id):
        if budget_type == BudgetType.MODEL_TRIAL_COUNT:
            max_trials = budget_amount 
            completed_trials = self._db.get_completed_trials_of_train_job(train_job_id)
            model_completed_trials = [x for x in completed_trials if x.model_id == model_id]
            return len(model_completed_trials) >= max_trials
        else:
            raise InvalidBudgetTypeException()

    def _read_worker_info(self):
        worker = self._db.get_train_job_worker(self._service_id)

        if worker is None:
            raise InvalidWorkerException()

        model = self._db.get_model(worker.model_id)
        train_job = self._db.get_train_job(worker.train_job_id)

        if model is None:
            raise InvalidModelException()

        if train_job is None:
            raise InvalidTrainJobException()

        return (
            train_job.budget_type, 
            train_job.budget_amount, 
            worker.model_id,
            model.model_serialized,
            train_job.id,
            train_job.train_dataset_uri,
            train_job.test_dataset_uri
        )

    def _make_client(self):
        admin_host = os.environ['ADMIN_HOST']
        admin_port = os.environ['ADMIN_PORT']
        advisor_host = os.environ['ADVISOR_HOST']
        advisor_port = os.environ['ADVISOR_PORT']
        superadmin_email = os.environ['SUPERADMIN_EMAIL']
        superadmin_password = os.environ['SUPERADMIN_PASSWORD']
        client = Client(admin_host=admin_host, 
                        admin_port=admin_port, 
                        advisor_host=advisor_host,
                        advisor_port=advisor_port)
        client.login(email=superadmin_email, password=superadmin_password)
        return client