from flask import Flask, request, jsonify
import os

from common import UserType

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
# Apps
####################################

@app.route('/apps', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_app(auth):
    params = get_request_params()
    return jsonify(admin.create_app(**params))

@app.route('/apps', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_apps(auth):
    params = get_request_params()
    return jsonify(admin.get_apps(**params))

@app.route('/apps/<app_name>', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_app(auth, app_name):
    params = get_request_params()
    return jsonify(admin.get_app(app_name, **params))

####################################
# Train Jobs
####################################

@app.route('/apps/<app_name>/train_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_train_job(auth, app_name):
    params = get_request_params()
    return jsonify(admin.create_train_job(app_name, **params))

@app.route('/apps/<app_name>/train_jobs', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_train_jobs(auth, app_name):
    params = get_request_params()
    return jsonify(admin.get_train_jobs(app_name, **params))

####################################
# Trials
####################################

@app.route('/apps/<app_name>/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_best_trials_by_app(auth, app_name):
    params = get_request_params()
    return jsonify(admin.get_best_trials_by_app(app_name, **params))

@app.route('/apps/<app_name>/train_jobs/<train_job_id>/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_trials(auth, app_name, train_job_id):
    params = get_request_params()
    return jsonify(admin.get_trials(app_name, train_job_id, **params))

@app.route('/apps/<app_name>/trials/<trial_id>/predict', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def predict_with_trial(auth, app_name, trial_id):
    params = get_request_params()
    preds = admin.predict_with_trial(app_name, trial_id, **params)
    return jsonify([to_json_serializable(x) for x in preds])

####################################
# Models
####################################

@app.route('/models', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER])
def get_models(auth):
    params = get_request_params()
    return jsonify(admin.get_models(**params))

@app.route('/models', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_model(auth):
    params = get_request_params()
    model_serialized = request.files['model_serialized'].read()
    params['model_serialized'] = model_serialized
    return jsonify(admin.create_model(**params))
