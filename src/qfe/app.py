from flask import Flask, jsonify, request
from .query_frontend import QueryFrontend

app = Flask(__name__)
qfe = QueryFrontend()

@app.route('/')
def index():
    return 'Query Frontend is up.'

@app.route('/predict', methods=['POST'])
def predict():
    params = request.get_json()
    input = params['input']
    type = params['type']
    
    #TODO: check input type
    result = qfe.predict(input)
    return jsonify(result)
