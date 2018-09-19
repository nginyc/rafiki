import time
import logging
import os
import traceback
import pprint

from .Advisor import Advisor
from .Proposal import Proposal

class InvalidAdvisorException(Exception):
    pass

class InvalidProposalException(Exception):
    pass

logger = logging.getLogger(__name__)

class AdvisorStore(object):
    def __init__(self):
        self._advisors = {}
        self._proposals = {}

    def create_advisor(self, advisor_inst, knob_config):
        advisor = Advisor(advisor_inst, knob_config)
        self._advisors[advisor.id] = advisor
        return advisor

    def get_advisor(self, advisor_id):
        print(advisor_id)
        print(self._advisors)
        if advisor_id not in self._advisors:
            raise InvalidAdvisorException()

        advisor = self._advisors[advisor_id]
        return advisor

    def update_advisor(self, advisor, advisor_inst):
        advisor.advisor_inst = advisor_inst
        return advisor

    def delete_advisor(self, advisor):
        del self._advisors[advisor.id]

    def add_proposal(self, advisor, knobs):
        proposal = Proposal(knobs)
        advisor.proposal_ids.append(proposal.id)
        
        self._proposals[proposal.id] = proposal

        return proposal

    def get_proposal(self, advisor_id, proposal_id):
        if advisor_id not in self._advisors:
            raise InvalidAdvisorException()

        advisor = self._advisors[advisor_id]

        if proposal_id not in advisor.proposal_ids:
            raise InvalidProposalException()

        if proposal_id not in self._proposals:
            raise InvalidProposalException()

        proposal = self._proposals[proposal_id]
        return proposal

    def get_proposals(self, advisor_id):
        if advisor_id not in self._advisors:
            raise InvalidAdvisorException()

        advisor = self._advisors[advisor_id]
        proposals = [self._proposals[x] for x in advisor.proposal_ids]
        return proposals

    def update_proposal(self, proposal, result_score):
        proposal.result_score = result_score
        return proposal


    