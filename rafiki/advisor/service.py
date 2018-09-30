import time
import logging
import os
import traceback
import pprint
import numpy as np

from rafiki.constants import AdvisorType

from .store import Store
from .btb_gp_advisor import BtbGpAdvisor

logger = logging.getLogger(__name__)

class InvalidAdvisorTypeException(Exception):
    pass

class InvalidAdvisorException(Exception):
    pass

class InvalidProposalException(Exception):
    pass

class AdvisorService(object):
    def __init__(self):
        self._store = Store()

    def create_advisor(self, knob_config, advisor_id=None):
        is_created = False
        advisor = None

        if advisor_id is not None:
            advisor = self._store.get_advisor(advisor_id)
            
        if advisor is None:
            advisor_inst = self._make_advisor(knob_config)
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
            # Whether the advisor has been deleted (maybe it already has been deleted)
            'is_deleted': is_deleted 
        }

    def generate_proposal(self, advisor_id):
        advisor = self._store.get_advisor(advisor_id)
        knobs = self._generate_proposal(advisor)

        return {
            'knobs': knobs
        }

    # Feedbacks to the advisor on the score of a set of knobs
    # Additionally, returns another proposal of knobs after ingesting feedback
    def feedback(self, advisor_id, knobs, score):
        advisor = self._store.get_advisor(advisor_id)

        if advisor is None:
            raise InvalidAdvisorException()

        advisor_inst = advisor.advisor_inst
        advisor_inst.feedback(knobs, score)
        knobs = self._generate_proposal(advisor)

        return {
            'knobs': knobs
        }

    def _generate_proposal(self, advisor):
        knobs = advisor.advisor_inst.propose()

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        return knobs

    def _simplify_value(self, value):
        # TODO: Support int64 & other non-serializable data formats
        if isinstance(value, np.int64):
            return int(value)

        return value

    def _make_advisor(self, knob_config, advisor_type=AdvisorType.BTB_GP):
        if advisor_type == AdvisorType.BTB_GP:
            return BtbGpAdvisor(knob_config)
        else:
            raise InvalidAdvisorTypeException()