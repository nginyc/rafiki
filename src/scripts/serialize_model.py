import importlib.util
import os
import sys

from admin import Admin
import model
from common import serialize_model

if len(sys.argv) < 2:
    print('Usage: python {} <model_class_name>'.format(__file__))
    exit(1)

model_class_name = sys.argv[1]

exec('from model.{} import {}'.format(model_class_name, model_class_name))

admin = Admin()
model_class = eval(model_class_name)
model_inst = model_class()
model_serialized = serialize_model(model_inst)

model_dir_path = os.path.abspath(os.path.dirname(model.__file__))
out_file_path = os.path.join(model_dir_path, '{}.pickle'.format(model_class_name))

with open(out_file_path, 'wb') as f:
    f.write(model_serialized)

print('Serialized model file saved at {}'.format(out_file_path))
