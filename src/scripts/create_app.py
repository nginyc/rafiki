from admin import Admin
from common import build_tf_keras_dataset_config

admin = Admin()

train_dataset_config = build_tf_keras_dataset_config(
    dataset_name='fashion_mnist', 
    train_or_test='train'
)
test_dataset_config = build_tf_keras_dataset_config(
    dataset_name='fashion_mnist', 
    train_or_test='test'
)

admin.create_app(
    name='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_config=train_dataset_config,
    test_dataset_config=test_dataset_config
)
