import uuid
import time
import logging
import os
import traceback
import pprint

class Advisor(object):
    def __init__(self, advisor_inst, knob_config, advisor_id=None):
        if advisor_id is not None:
            self.id = advisor_id
        else:
            self.id = str(uuid.uuid4())
        
        self.advisor_inst = advisor_inst
        self.knob_config = knob_config

class Store(object):
    def __init__(self):
        self._advisors = {}

    def create_advisor(self, advisor_inst, knob_config, advisor_id=None):
        advisor = Advisor(advisor_inst, knob_config, advisor_id)
        self._advisors[advisor.id] = advisor
        return advisor

    def get_advisor(self, advisor_id):
        if advisor_id not in self._advisors:
            return None

        advisor = self._advisors[advisor_id]
        return advisor

    def update_advisor(self, advisor, advisor_inst):
        advisor.advisor_inst = advisor_inst
        return advisor

    def delete_advisor(self, advisor):
        del self._advisors[advisor.id]
