from flask import Flask, request, jsonify
import os
import traceback

from rafiki.constants import UserType
from rafiki.utils.auth import generate_token, decode_token, UnauthorizedException, auth
from rafiki.utils.parse import get_request_params

from .AdvisorService import AdvisorService

service = AdvisorService()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Rafiki Advisor is up.'

@app.route('/advisors', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_advisor(auth):
    params = get_request_params()
    return jsonify(service.create_advisor(**params))

@app.route('/advisors/<advisor_id>/propose', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def generate_proposal(auth, advisor_id):
    params = get_request_params()
    return jsonify(service.generate_proposal(advisor_id, **params))

@app.route('/advisors/<advisor_id>/proposals/<proposal_id>', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def set_result_of_proposal(auth, advisor_id, proposal_id):
    params = get_request_params()
    return jsonify(service.set_result_of_proposal(advisor_id, proposal_id, **params))

@app.route('/advisors/<advisor_id>', methods=['DELETE'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def delete_advisor(auth, advisor_id):
    params = get_request_params()
    return jsonify(service.delete_advisor(advisor_id, **params))

# Handle uncaught exceptions with a server error & the error's stack trace (for development)
@app.errorhandler(Exception)
def handle_error(error):
    return traceback.format_exc(), 500