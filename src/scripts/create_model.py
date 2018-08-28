from admin import Admin
from common import serialize_model
from model.SingleHiddenLayerTensorflowModel import SingleHiddenLayerTensorflowModel

admin = Admin()
model = SingleHiddenLayerTensorflowModel()
model_serialized = serialize_model(model)

admin.create_model(
    name='single_hidden_layer_tf',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    model_serialized=model_serialized
)
