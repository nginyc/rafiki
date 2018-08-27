from flask import Flask, request, jsonify
import os

from .Admin import Admin

admin = Admin(
  db_host=os.environ['MYSQL_HOST'],
  db_port=os.environ['MYSQL_PORT'],
  db_password=os.environ['MYSQL_PASSWORD'],
  db_username=os.environ['MYSQL_USER'],
  db_database=os.environ['MYSQL_DATABASE']
)

app = Flask(__name__)

@app.route('/')
def index():
  return 'Admin is up.'


@app.route('/start')
def start():
  admin.start()
  return jsonify({ 'success': True })


@app.route('/stop')
def stop():
  admin.stop()
  return jsonify({ 'success': True })


@app.route("/dataruns", methods=['POST'])
def create_datarun():
  params = request.get_json()
  
  return jsonify(admin.create_datarun(
    dataset_name=params['dataset_name'],
    preparator_type=params['preparator_type'],
    preparator_params=params['preparator_params'],
    budget_type=params['budget_type'],
    budget=params['budget']
  ))

@app.route("/dataruns/<datarun_id>", methods=['GET'])
def get_datarun(datarun_id):
  return jsonify(admin.get_datarun(
    datarun_id=datarun_id
  ))


@app.route("/datasets/<dataset_id>", methods=['GET'])
def get_dataset(dataset_id):
  return jsonify(admin.get_dataset(
    dataset_id=dataset_id
  ))

@app.route("/datasets/<dataset_id>/random", methods=['GET'])
@app.route("/datasets/<dataset_id>/<int:example_id>", methods=['GET'])
def get_dataset_example(dataset_id, example_id=None):
  return jsonify(admin.get_dataset_example(
    dataset_id=dataset_id,
    example_id=example_id
  ))


@app.route("/models/<model_id>", methods=['GET'])
def get_model(model_id):
  return jsonify(admin.get_model(
    model_id=model_id
  ))


@app.route("/models/<model_id>/queries", methods=['POST'])
def query_model(model_id):
  params = request.get_json()
  return jsonify(admin.query_model(
    model_id=model_id,
    queries=params['queries']
  ))

@app.route("/models/<model_id>/deployments/<deployment_name>", methods=['POST'])
def create_model_deployment(model_id, deployment_name):
  return jsonify(admin.deploy_model(
    model_id=model_id,
    deployment_name=deployment_name
  ))

@app.route("/apps/<app_name>", methods=['POST'])
def create_app(app_name):
  params = request.get_json()
  return jsonify(admin.create_app(
    name=app_name,
    slo_micros=params['slo_micros'], 
    model_deployment_names=params['model_deployment_names']
  ))

@app.route("/apps/<app_name>", methods=['DELETE'])
def delete_app(app_name):
  return jsonify(admin.delete_app(
    name=app_name
  ))
