from flask import Flask, request, jsonify

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
def create_user():
    params = get_request_params()
    return jsonify(admin.create_user(**params))

####################################
# Apps
####################################

@app.route('/apps', methods=['POST'])
def create_app():
    params = get_request_params()
    return jsonify(admin.create_app(**params))

@app.route('/apps', methods=['GET'])
def get_apps():
    params = get_request_params()
    return jsonify(admin.get_apps(**params))

@app.route('/apps/<app_name>', methods=['GET'])
def get_app(app_name):
    params = get_request_params()
    return jsonify(admin.get_app(app_name, **params))

####################################
# Train Jobs
####################################

@app.route('/apps/<app_name>/train_jobs', methods=['POST'])
def create_train_job(app_name):
    params = get_request_params()
    return jsonify(admin.create_train_job(app_name, **params))

@app.route('/apps/<app_name>/train_jobs', methods=['GET'])
def get_train_jobs(app_name):
    params = get_request_params()
    return jsonify(admin.get_train_jobs(app_name, **params))

####################################
# Trials
####################################

@app.route('/apps/<app_name>/trials', methods=['GET'])
def get_best_trials_by_app(app_name):
    params = get_request_params()
    return jsonify(admin.get_best_trials_by_app(app_name, **params))

@app.route('/apps/<app_name>/train_jobs/<train_job_id>/trials', methods=['GET'])
def get_trials(app_name, train_job_id):
    params = get_request_params()
    return jsonify(admin.get_trials(app_name, train_job_id, **params))

@app.route('/apps/<app_name>/trials/<trial_id>/predict', methods=['POST'])
def predict_with_trial(app_name, trial_id):
    params = get_request_params()
    preds = admin.predict_with_trial(app_name, trial_id, **params)
    return jsonify([to_json_serializable(x) for x in preds])

####################################
# Models
####################################

@app.route('/models', methods=['GET'])
def get_models():
    params = get_request_params()
    return jsonify(admin.get_models(**params))

@app.route('/models', methods=['POST'])
def create_model():
    params = get_request_params()
    model_serialized = request.files['model_serialized'].read()
    params['model_serialized'] = model_serialized

    return jsonify(admin.create_model(**params))

