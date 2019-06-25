from typing import Union
import uuid

class Query():
    def __init__(self, query: any):
        self.id = str(uuid.uuid4())
        self.query = query

class Prediction():
    def __init__(self, 
                # Raw prediction, or None if the worker is unable to make a prediction (e.g. errored)
                prediction: Union[any, None], 
                # ID of query of prediction
                query_id: str, 
                # Worker who made the prediction, if any
                worker_id: str = None): 
        self.prediction = prediction
        self.query_id = query_id
        self.worker_id = worker_id