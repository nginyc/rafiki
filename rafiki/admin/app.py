from flask import Flask, request, jsonify
import os
import traceback

from rafiki.constants import UserType
from .auth import generate_token, decode_token, UnauthorizedException, auth
from .parse import get_request_params, to_json_serializable
from .Admin import Admin

admin = Admin()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Rafiki Admin is up.'

####################################
# Users
####################################

@app.route('/users', methods=['POST'])
@auth([UserType.ADMIN])
def create_user(auth):
    params = get_request_params()
    return jsonify(admin.create_user(**params))

@app.route('/tokens', methods=['POST'])
def generate_user_token():
    params = get_request_params()

    # Error will be thrown here if credentials are invalid
    user = admin.authenticate_user(**params)

    auth = {
        'user_id': user['id'],
        'user_type': user['user_type']
    }
    
    token = generate_token(auth)

    return jsonify({
        'user_id': user['id'],
        'user_type': user['user_type'],
        'token': token
    })

####################################
# Train Jobs
####################################

@app.route('/train_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_train_job(auth):
    params = get_request_params()
    return jsonify(admin.create_train_job(auth['user_id'], **params))

@app.route('/train_jobs', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_train_jobs_of_app(auth):
    params = get_request_params()
    return jsonify(admin.get_train_jobs_of_app(**params))

@app.route('/train_jobs/<train_job_id>', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_train_job(auth, train_job_id):
    params = get_request_params()
    return jsonify(admin.get_train_job(train_job_id, **params))

@app.route('/train_jobs/<train_job_id>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def stop_train_job(auth, train_job_id):
    params = get_request_params()
    return jsonify(admin.stop_train_job(train_job_id, **params))

@app.route('/train_job_services/<service_id>/stop', methods=['POST'])
@auth([])
def stop_train_job_service(auth, service_id):
    params = get_request_params()
    return jsonify(admin.stop_train_job_service(service_id, **params))

####################################
# Inference Jobs
####################################

@app.route('/inference_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_inference_jobs(auth):
    params = get_request_params()
    return jsonify(admin.create_inference_job(auth['user_id'], **params))

@app.route('/inference_jobs', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_inference_jobs(auth, app):
    params = get_request_params()
    return jsonify(admin.get_inference_jobs(app, **params))

####################################
# Trials
####################################

@app.route('/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_trials_of_app(auth):
    params = get_request_params()

    # Return best trials by app
    if params.get('type') == 'best':
        del params['type']

        if 'max_count' in params:
            params['max_count'] = int(params['max_count'])

        return jsonify(admin.get_best_trials_of_app(**params))
    
    # Return all trials by app
    else:
        return jsonify(admin.get_trials_of_app(**params))


@app.route('/train_job/<train_job_id>/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_trials_of_train_job(auth, train_job_id):
    params = get_request_params()
    return jsonify(admin.get_trials_of_train_job(train_job_id, **params))

    
@app.route('/trials/<trial_id>/predict', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def predict_with_trial(auth, trial_id):
    params = get_request_params()
    preds = admin.predict_with_trial(trial_id, **params)
    return jsonify([to_json_serializable(x) for x in preds])

####################################
# Models
####################################

@app.route('/models', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_model(auth):
    params = get_request_params()
    model_serialized = request.files['model_serialized'].read()
    params['model_serialized'] = model_serialized
    return jsonify(admin.create_model(auth['user_id'], **params))

@app.route('/models', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER])
def get_models(auth):
    params = get_request_params()

    # Return models by task
    if params.get('task') is not None:
        return jsonify(admin.get_models_of_task(**params))
    
    # Return all models
    else:
        return jsonify(admin.get_models(**params))

    return jsonify(admin.get_models(**params))

# Handle uncaught exceptions with a server error & the error's stack trace (for development)
@app.errorhandler(Exception)
def handle_error(error):
    return traceback.format_exc(), 500