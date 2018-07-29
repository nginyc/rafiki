from flask import Flask, request
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

@app.route("/datarun", methods=['POST'])
def post_datarun():
  params = request.get_json()
  return str(admin.create_datarun(**params))

@app.route("/datarun", methods=['GET'])
def get_datarun():
  params = request.get_json()
  return str(admin.get_datarun(**params))