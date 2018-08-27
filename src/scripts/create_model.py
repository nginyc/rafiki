from admin import Admin
from model import SingleHiddenLayerTensorflowModel

admin = Admin()

admin.create_model(
    name='single_hidden_layer_tf',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    model_inst=SingleHiddenLayerTensorflowModel()
)
