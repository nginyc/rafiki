import os
from flask import Flask, jsonify, request, g
import threading

from .predictor import Predictor

service_id = os.environ['RAFIKI_SERVICE_ID']

app = Flask(__name__)

def get_predictor():
    # Allow multiple threads to each have their own instance of predictor
    if not hasattr(g, 'predictor'):
        g.predictor = Predictor(service_id)
        g.predictor.start()
    
    return g.predictor

@app.route('/')
def index():
    return 'Predictor is up.'

@app.route('/predict', methods=['POST'])
def predict():
    predictor = get_predictor()
    params = request.get_json()
    query = params['query']
    
    #TODO: check input type
    result = predictor.predict(query)
    return jsonify(result)
