from admin import Admin
from common import TfKerasDatasetConfig

admin = Admin()

admin.create_app(
    name='name',
    task='image_classification',
    train_dataset_config=TfKerasDatasetConfig(dataset_name='fashion_mnist')
)
