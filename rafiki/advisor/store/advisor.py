import uuid

class Advisor(object):
    def __init__(self, advisor_inst, knob_config, advisor_id=None):
        if advisor_id is not None:
            self.id = advisor_id
        else:
            self.id = str(uuid.uuid4())
        
        self.advisor_inst = advisor_inst
        self.knob_config = knob_config
        self.proposal_ids = []