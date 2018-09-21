import time
import logging
import os
import traceback
import pprint
import numpy as np

from .store import AdvisorStore
from .advisors import BtbGpAdvisor

logger = logging.getLogger(__name__)

class InvalidAdvisorException(Exception):
    pass

class InvalidProposalException(Exception):
    pass

class AdvisorService(object):
    def __init__(self):
        self._store = AdvisorStore()

    def create_advisor(self, knob_config, advisor_id=None):
        is_created = False
        advisor = None

        if advisor_id is not None:
            advisor = self._store.get_advisor(advisor_id)
            
        if advisor is None:
            advisor_inst = BtbGpAdvisor(knob_config)
            advisor = self._store.create_advisor(advisor_inst, knob_config, advisor_id)
            is_created = True

        return {
            'id': advisor.id,
            'is_created': is_created # Whether a new advisor has been created
        }

    def delete_advisor(self, advisor_id):
        is_deleted = False

        advisor = self._store.get_advisor(advisor_id)

        if advisor is not None:
            self._store.delete_advisor(advisor)
            is_deleted = True

        return {
            'id': advisor_id,
            'is_deleted': is_deleted # Whether the advisor has been deleted (maybe it already has been deleted)
        }

    def generate_proposal(self, advisor_id):
        advisor = self._store.get_advisor(advisor_id)
        advisor_inst = advisor.advisor_inst
        knobs = advisor_inst.propose()

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        proposal = self._store.add_proposal(advisor, knobs)
        
        return {
            'knobs': knobs,
            'id': proposal.id
        }

    def set_result_of_proposal(self, advisor_id, proposal_id, score):
        advisor = self._store.get_advisor(advisor_id)

        if advisor is None:
            raise InvalidAdvisorException()

        proposal = self._store.get_proposal(advisor_id, proposal_id)

        if proposal is None:
            raise InvalidProposalException()

        advisor_inst = advisor.advisor_inst
        advisor_inst.set_result_of_proposal(proposal.knobs, score)

        proposal = self._store.update_proposal(proposal, score)
        return {
            'proposal_id': proposal.id
        }

    def _simplify_value(self, value):
        # TODO: Support int64 & other non-serializable data formats
        if isinstance(value, np.int64):
            return int(value)

        return value
    