import os
from flask import Flask, jsonify, request

from .predictor import Predictor

service_id = os.environ['RAFIKI_SERVICE_ID']

app = Flask(__name__)
predictor = Predictor(service_id)
predictor.start()

@app.route('/')
def index():
    return 'Predictor is up.'

@app.route('/predict', methods=['POST'])
def predict():
    params = request.get_json()
    query = params['query']
    
    #TODO: check input type
    result = predictor.predict(query)
    return jsonify(result)
