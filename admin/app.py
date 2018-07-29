from flask import Flask, request, jsonify
import os

from .Admin import Admin

admin = Admin(
  host=os.environ['MYSQL_HOST'],
  port=os.environ['MYSQL_PORT'],
  password=os.environ['MYSQL_PASSWORD'],
  username=os.environ['MYSQL_USER'],
  database=os.environ['MYSQL_DATABASE']
)

app = Flask(__name__)

@app.route('/')
def index():
  return 'Admin is up.'

@app.route("/dataruns", methods=['POST'])
def post_datarun():
  params = request.get_json()
  
  return jsonify(admin.create_datarun(
    dataset_url=params['dataset_url'],
    class_column=params['class_column'],
    budget_type=params['budget_type'],
    budget=params['budget']
  ))

@app.route("/dataruns/<datarun_id>", methods=['GET'])
def get_datarun(datarun_id):
  return jsonify(admin.get_datarun(
    datarun_id=datarun_id
  ))

@app.route("/classifiers/<classifier_id>", methods=['GET'])
def get_classifier(classifier_id):
  return jsonify(admin.get_classifier(
    classifier_id=classifier_id
  ))

@app.route("/classifiers/<classifier_id>/queries", methods=['POST'])
def query_classifier(classifier_id):
  params = request.get_json()
  return jsonify(admin.query_classifier(
    classifier_id=classifier_id,
    queries=params['queries']
  ))