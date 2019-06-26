import os
import logging
from flask import Flask, jsonify, request, g

from .predictor import Predictor

service_id = os.environ['RAFIKI_SERVICE_ID']

logger = logging.getLogger(__name__)
app = Flask(__name__)

class InvalidQueryFormatError(Exception): pass

def get_predictor() -> Predictor:
    # Allow multiple threads to each have their own instance of predictor
    if not hasattr(g, 'predictor'):
        g.predictor = Predictor(service_id)
    
    return g.predictor

# Extract request params from Flask request
def get_request_params():
    # Get params from body as JSON
    params = request.get_json()

    # If the above fails, get params from body as form data
    if params is None:
        params = request.form.to_dict()

    # Merge in query params
    query_params = {
        k: v
        for k, v in request.args.items()
    }
    params = {**params, **query_params}

    return params

@app.route('/')
def index():
    return 'Predictor is up.'

@app.route('/predict', methods=['POST'])
def predict():
    predictor = get_predictor()
    params = get_request_params()

    # Must either have `query` or `queries` key
    if not 'query' in params and not 'queries' in params:
        raise InvalidQueryFormatError('Must have either `query` or `queries` attribute')

    # Either do single prediction or bulk predictions 
    if 'queries' in params:
        predictions = predictor.predict(params['queries'])
        return jsonify({
            'prediction': None,
            'predictions': predictions
        })
    else:
        predictions = predictor.predict([params['query']])
        assert len(predictions) == 1
        return jsonify({
            'prediction': predictions[0],
            'predictions': []
        })


