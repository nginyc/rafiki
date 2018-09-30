import uuid

class Proposal(object):
    def __init__(self, knobs):
        self.id = str(uuid.uuid4())
        self.knobs = knobs
        self.result_score = None