from rafiki.client import Client
from rafiki.constants import DatasetTask

def main():
    client = Client()
    client.login(email='superadmin@rafiki', password='rafiki')

    client.create_model(
        name='multi_layer_perceptron',
        task=DatasetTask.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/multi_layer_perceptron.py',
        model_class='MultiLayerPerceptron'
    )

    client.create_train_job(
        app='fashion_mnist_app',
        task=DatasetTask.IMAGE_CLASSIFICATION,
        train_dataset_uri='https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_train.zip?raw=true',
        test_dataset_uri='https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_test.zip?raw=true',
        budget_type='MODEL_TRIAL_COUNT',
        budget_amount=3
    )

if __name__ == '__main__':
    main()