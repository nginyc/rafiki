from admin import Admin

admin = Admin()

admin.create_app(
    name='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
    test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test'
)
