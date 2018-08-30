import sys

from admin import Admin
from common import serialize_model
from model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel

if len(sys.argv) < 3:
    print('Usage: python {} <email> <password>'.format(__file__))
    exit(1)

email = sys.argv[1]
password = sys.argv[2]

admin = Admin()

model = SingleHiddenLayerTensorflowModel()
model_serialized = serialize_model(model)

user = admin.authenticate_user(email, password)

admin.create_model(
    user_id=user['id'],
    name='single_hidden_layer_tf',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    model_serialized=model_serialized
)
