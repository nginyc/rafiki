import os
from flask import Flask, jsonify, request

from .QueryFrontend import QueryFrontend

service_id = os.environ['RAFIKI_SERVICE_ID']

app = Flask(__name__)
query_frontend = QueryFrontend(service_id)

@app.route('/')
def index():
    return 'Query Frontend is up.'

@app.route('/predict', methods=['POST'])
def predict():
    params = request.get_json()
    query = params['query']
    query_type = params['type']
    
    #TODO: check input type
    result = query_frontend.predict(query)
    return jsonify(result)
