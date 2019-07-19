#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from flask import Flask, request, jsonify, g, make_response
from flask_cors import CORS
import traceback
import json
import tempfile
import requests
from datetime import datetime
import pickle

from rafiki.constants import UserType
from rafiki.utils.auth import generate_token, auth, UnauthorizedError

from .admin import Admin

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Rafiki Admin is up.'

####################################
# Users
####################################

@app.route('/users', methods=['POST'])
@auth([UserType.ADMIN])
def create_user(auth):
    admin = get_admin()
    params = get_request_params()

    # Only superadmins can create admins
    if auth['user_type'] != UserType.SUPERADMIN and \
            params['user_type'] in [UserType.ADMIN, UserType.SUPERADMIN]:
        raise UnauthorizedError()

    with admin:
        return jsonify(admin.create_user(**params))

@app.route('/users', methods=['GET'])
@auth([UserType.ADMIN])
def get_users(auth):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_users(**params))

@app.route('/users', methods=['DELETE'])
@auth([UserType.ADMIN])
def ban_user(auth):
    admin = get_admin()
    params = get_request_params()

    with admin:
        user = admin.get_user_by_email(params['email'])
        
        if user is not None:
            # Only superadmins can ban admins
            if auth['user_type'] != UserType.SUPERADMIN and \
                    user['user_type'] in [UserType.ADMIN, UserType.SUPERADMIN]:
                raise UnauthorizedError()

            # Cannot ban yourself
            if auth['user_id'] == user['id']:
                raise UnauthorizedError()
        
        return jsonify(admin.ban_user(**params))

@app.route('/tokens', methods=['POST'])
def generate_user_token():
    admin = get_admin()
    params = get_request_params()

    # Error will be thrown here if credentials are invalid
    with admin:
        user = admin.authenticate_user(**params)

    # User cannot be banned
    if user.get('banned_date') is not None and datetime.now() > user.get('banned_date'):
        raise UnauthorizedError('User is banned')
    
    token = generate_token(user)

    return jsonify({
        'user_id': user['id'],
        'user_type': user['user_type'],
        'token': token
    })

####################################
# Datasets
####################################

@app.route('/datasets', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def create_dataset(auth):
    admin = get_admin()
    params = get_request_params()

    # Temporarily store incoming dataset data as file
    with tempfile.NamedTemporaryFile() as f:
        if 'dataset' in request.files:
            # Save dataset data in request body
            file_storage = request.files['dataset']
            file_storage.save(f.name)
            file_storage.close()
        else:
            # Download dataset at URL and save it
            assert 'dataset_url' in params
            r = requests.get(params['dataset_url'], allow_redirects=True)
            f.write(r.content)
            del params['dataset_url']

        params['data_file_path'] = f.name

        with admin:
            return jsonify(admin.create_dataset(auth['user_id'], **params))

@app.route('/datasets', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_datasets(auth):
    admin = get_admin()
    params = get_request_params()
    with admin:
        return jsonify(admin.get_datasets(auth['user_id'], **params))

####################################
# Train Jobs
####################################

@app.route('/train_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def create_train_job(auth):
    admin = get_admin()
    params = get_request_params()

    with admin:
        # Ensure that datasets are owned by current user
        dataset_attrs = ['train_dataset_id', 'val_dataset_id']
        for attr in dataset_attrs:
            if attr in params:
                dataset_id = params[attr]
                dataset = admin.get_dataset(dataset_id)
                if auth['user_id'] != dataset['owner_id']:
                    raise UnauthorizedError('You have no access to dataset of ID "{}"'.format(dataset_id))
        
        return jsonify(admin.create_train_job(auth['user_id'], **params))

@app.route('/train_jobs', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_jobs_by_user(auth):
    admin = get_admin()
    params = get_request_params()

    assert 'user_id' in params

    # Non-admins can only get their own jobs
    if auth['user_type'] in [UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER] \
            and auth['user_id'] != params['user_id']:
        raise UnauthorizedError()

    with admin:
        return jsonify(admin.get_train_jobs_by_user(**params))

@app.route('/train_jobs/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_jobs_by_app(auth, app):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_train_jobs_by_app(auth['user_id'], app, **params))

@app.route('/train_jobs/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_train_job(auth['user_id'], app, app_version=int(app_version), **params))

@app.route('/train_jobs/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def stop_train_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.stop_train_job(auth['user_id'], app, app_version=int(app_version), **params))

####################################
# Trials
####################################

@app.route('/trials/<trial_id>/logs', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial_logs(auth, trial_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_trial_logs(trial_id, **params))

@app.route('/trials/<trial_id>/parameters', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial_parameters(auth, trial_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        trial_params = admin.get_trial_parameters(trial_id, **params)
 
    trial_params = pickle.dumps(trial_params) # Pickle to convert to bytes
    res = make_response(trial_params)
    res.headers.set('Content-Type', 'application/octet-stream')
    return res

@app.route('/trials/<trial_id>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial(auth, trial_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_trial(trial_id, **params))

@app.route('/train_jobs/<app>/<app_version>/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trials_of_train_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    # Return best trials by train job
    if params.get('type') == 'best':
        del params['type']

        if 'max_count' in params:
            params['max_count'] = int(params['max_count'])

        with admin:
            return jsonify(admin.get_best_trials_of_train_job(auth['user_id'], app, app_version=int(app_version), **params))
    
    # Return all trials by train job
    else:
        with admin:
            return jsonify(admin.get_trials_of_train_job(auth['user_id'], app, app_version=int(app_version), **params))

####################################
# Inference Jobs
####################################

@app.route('/inference_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def create_inference_job(auth):
    admin = get_admin()
    params = get_request_params()

    if 'app_version' in params:
        params['app_version'] = int(params['app_version'])

    with admin:
        return jsonify(admin.create_inference_job(auth['user_id'], **params))

@app.route('/inference_jobs', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_inference_jobs_by_user(auth):
    admin = get_admin()
    params = get_request_params()

    assert 'user_id' in params

    # Non-admins can only get their own jobs
    if auth['user_type'] in [UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER] \
            and auth['user_id'] != params['user_id']:
        raise UnauthorizedError()

    with admin:
        return jsonify(admin.get_inference_jobs_by_user(**params))

@app.route('/inference_jobs/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_inference_jobs_of_app(auth, app):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_inference_jobs_of_app(auth['user_id'], app, **params))

@app.route('/inference_jobs/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_running_inference_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_running_inference_job(auth['user_id'], app, app_version=int(app_version), **params))

@app.route('/inference_jobs/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def stop_inference_job(auth, app, app_version=-1):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.stop_inference_job(auth['user_id'], app, app_version=int(app_version), **params))

####################################
# Models
####################################

@app.route('/models', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_model(auth):
    admin = get_admin()
    params = get_request_params()

    # Expect model file as bytes
    model_file_bytes = request.files['model_file_bytes'].read()
    params['model_file_bytes'] = model_file_bytes

    # Expect model dependencies as dict
    if 'dependencies' in params and isinstance(params['dependencies'], str):
        params['dependencies'] = json.loads(params['dependencies'])

    with admin:
        return jsonify(admin.create_model(auth['user_id'], **params))

@app.route('/models/available', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_available_models(auth):
    admin = get_admin()
    params = get_request_params()
    with admin:
        return jsonify(admin.get_available_models(auth['user_id'], **params))

@app.route('/models/<model_id>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_model(auth, model_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        # Non-admins cannot access others' models
        if auth['user_type'] in [UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER]:
            model = admin.get_model(model_id)
            if auth['user_id'] != model['user_id']:
                raise UnauthorizedError()  
                
        return jsonify(admin.get_model(model_id, **params))

@app.route('/models/<model_id>', methods=['DELETE'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def delete_model(auth, model_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        # Non-admins cannot delete others' models
        if auth['user_type'] in [UserType.MODEL_DEVELOPER]:
            model = admin.get_model(model_id)
            if auth['user_id'] != model['user_id']:
                raise UnauthorizedError()  

        return jsonify(admin.delete_model(model_id, **params))

@app.route('/models/<model_id>/model_file', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def download_model_file(auth, model_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        # Non-admins cannot access others' models
        if auth['user_type'] in [UserType.MODEL_DEVELOPER]:
            model = admin.get_model(model_id)
            if auth['user_id'] != model['user_id']:
                raise UnauthorizedError()  


        model_file = admin.get_model_file(model_id, **params)

    res = make_response(model_file)
    res.headers.set('Content-Type', 'application/octet-stream')
    return res

####################################
# Administrative Actions
####################################

@app.route('/actions/stop_all_jobs', methods=['POST'])
@auth([])
def stop_all_jobs(auth):
    admin = get_admin()

    with admin:
        train_jobs = admin.stop_all_train_jobs()
        inference_jobs = admin.stop_all_inference_jobs()
        return jsonify({
            'train_jobs': train_jobs,
            'inference_jobs': inference_jobs
        })

####################################
# Internal Events
####################################

@app.route('/event/<name>', methods=['POST'])
@auth([])
def handle_event(auth, name):
    admin = get_admin()
    params = get_request_params()
    with admin:
        return jsonify(admin.handle_event(name, **params))


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

def get_admin():
    # Allow multiple threads to each have their own instance of admin
    if not hasattr(g, 'admin'):
        g.admin = Admin()
    
    return g.admin
