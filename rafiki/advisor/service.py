import time
import logging
import os
import uuid
import traceback
import pprint
from typing import Dict

from .advisor import make_advisor, BaseAdvisor

logger = logging.getLogger(__name__)

class InvalidAdvisorError(Exception): pass

class AdvisorService(object):
    def __init__(self):
        self._advisors: Dict[str, BaseAdvisor] = {}

    def create_advisor(self, knob_config, advisor_id=None, 
                        advisor_type=None, advisor_config={}):
        is_created = False
        advisor = None

        if advisor_id is not None:
            advisor = self._get_advisor(advisor_id)
            
        if advisor is None:
            advisor = make_advisor(knob_config, advisor_type=advisor_type, **advisor_config)
            advisor_id = str(uuid.uuid4()) if advisor_id is None else advisor_id
            self._advisors[advisor_id] = advisor
            is_created = True
            logger.info('Created advisor {} of ID "{}"...'.format(advisor.__class__, advisor_id))

        return {
            'id': advisor_id,
            'is_created': is_created # Whether a new advisor has been created
        }

    def delete_advisor(self, advisor_id):
        is_deleted = False
        advisor = self._get_advisor(advisor_id)

        if advisor is not None:
            del self._advisors[advisor_id]
            is_deleted = True
            logger.info('Deleted advisor of ID "{}"...'.format(advisor_id))

        return {
            'id': advisor_id,
            # Whether the advisor has been deleted (maybe it already has been deleted)
            'is_deleted': is_deleted 
        }

    def get_proposal_from_advisor(self, advisor_id, trial_no, 
                                total_trials, concurrent_trial_nos=[]):

        advisor = self._get_advisor(advisor_id)
        if advisor is None:
            raise InvalidAdvisorError()

        proposal = advisor.propose(trial_no, total_trials, concurrent_trial_nos)
        logger.info('[ID: "{}"] Proposing {} for trial #{} with concurrent trials {}...'
                    .format(advisor_id, proposal.to_jsonable(), trial_no, concurrent_trial_nos))
        return proposal

    def feedback(self, advisor_id, score, proposal):
        advisor = self._get_advisor(advisor_id)
        if advisor is None:
            raise InvalidAdvisorError()

        logger.info('[ID: "{}"] Received feedback of score {} for proposal {}...'
                    .format(advisor_id, score, proposal.to_jsonable()))

        advisor.feedback(score, proposal)

    def _get_advisor(self, advisor_id):
        if advisor_id not in self._advisors:
            return None

        advisor = self._advisors[advisor_id]
        return advisor
