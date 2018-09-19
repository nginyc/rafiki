import time
import logging
import os
import traceback
import pprint
import numpy as np

from .store import AdvisorStore
from .advisors import BtbGpAdvisor

logger = logging.getLogger(__name__)

class AdvisorService(object):
    def __init__(self):
        self._store = AdvisorStore()

    def create_advisor(self, knob_config):
        advisor_inst = BtbGpAdvisor(knob_config)
        advisor = self._store.create_advisor(advisor_inst, knob_config)
        return {
            'id': advisor.id
        }

    def delete_advisor(self, advisor_id):
        advisor = self._store.get_advisor(advisor_id)
        self._store.delete_advisor(advisor)
        return {
            'id': advisor.id
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
        advisor_inst = advisor.advisor_inst

        proposal = self._store.get_proposal(advisor_id, proposal_id)
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
    