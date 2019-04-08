from flask import Flask, request, jsonify
import os
import traceback
import json

from rafiki.model import deserialize_knob_config
from rafiki.constants import UserType
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.utils.auth import generate_token, UnauthorizedError, auth

from .service import AdvisorService, InvalidAdvisorError
from .advisor import Proposal

service = AdvisorService()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Rafiki Advisor is up.'

@app.route('/tokens', methods=['POST'])
def generate_user_token():
    params = get_request_params()

    # Only superadmin can authenticate (other users must use Rafiki Admin)
    if not (params['email'] == SUPERADMIN_EMAIL and \
            params['password'] == SUPERADMIN_PASSWORD):
        raise UnauthorizedError()
    
    auth = {
        'user_type': UserType.SUPERADMIN
    }
    
    token = generate_token(auth)

    return jsonify({
        'user_type': auth['user_type'],
        'token': token
    })

@app.route('/advisors', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_advisor(auth):
    params = get_request_params()

    # Deserialize knob config
    knob_config = deserialize_knob_config(params['knob_config_str'])
    del params['knob_config_str']

    return jsonify(service.create_advisor(knob_config, **params))

@app.route('/advisors/<advisor_id>/propose', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_proposal_from_advisor(auth, advisor_id):
    params = get_request_params()
    proposal = service.get_proposal_from_advisor(advisor_id, **params)
    return jsonify(proposal.to_jsonable())

@app.route('/advisors/<advisor_id>/feedback', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def feedback(auth, advisor_id):
    params = get_request_params()

    if 'proposal' in params:
        params['proposal'] = Proposal.from_jsonable(params['proposal'])

    return jsonify(service.feedback(advisor_id, **params))

@app.route('/advisors/<advisor_id>', methods=['DELETE'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def delete_advisor(auth, advisor_id):
    params = get_request_params()
    return jsonify(service.delete_advisor(advisor_id, **params))

@app.errorhandler(InvalidAdvisorError)
def handle_invalid_advisor_error(error):
    return jsonify({
        'error': True,
        'message': 'Invalid advisor.'
    }), 400

# Handle uncaught exceptions with a server error & the error's stack trace (for development)
@app.errorhandler(Exception)
def handle_error(error):
    return traceback.format_exc(), 500

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