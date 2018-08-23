from admin import Admin
from common import TfKerasDatasetConfig

admin = Admin()

train_dataset_config = TfKerasDatasetConfig(dataset_name='fashion_mnist', train_or_test='train')
test_dataset_config = TfKerasDatasetConfig(dataset_name='fashion_mnist', train_or_test='test')

admin.create_app(
    name='fashion_mnist_app',
    task='image_classification',
    train_dataset_config=train_dataset_config,
    test_dataset_config=test_dataset_config
)
