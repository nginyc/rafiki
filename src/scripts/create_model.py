from admin import Admin
from common import TfKerasDatasetConfig
from model import SingleHiddenLayerTensorflowModel

admin = Admin()

admin.create_model(
    name='single_hidden_layer_tf',
    task='image_classification',
    model_inst=SingleHiddenLayerTensorflowModel()
)
