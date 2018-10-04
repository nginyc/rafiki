from rafiki.client import Client
from rafiki.constants import DatasetTask

def create_model(client):
    model_data = client.create_model(
        name='multi_layer_perceptron',
        task=DatasetTask.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/multi_layer_perceptron.py',
        model_class='MultiLayerPerceptron'
    )
    print(model_data)

def create_train_job(client):
    train_job_data = client.create_train_job(
        app='fashion_mnist_app',
        task=DatasetTask.IMAGE_CLASSIFICATION,
        train_dataset_uri='https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_train.zip?raw=true',
        test_dataset_uri='https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_test.zip?raw=true',
        budget_type='MODEL_TRIAL_COUNT',
        budget_amount=3
    )
    print(train_job_data)

def get_best_trials_of_train_job(client):
    best_trials = client.get_best_trials_of_train_job(app='fashion_mnist_app')
    print(best_trials)

def main():
    client = Client()
    client.login(email='superadmin@rafiki', password='rafiki')

    # create_model(client)
    # create_train_job(client)
    get_best_trials_of_train_job(client)

if __name__ == '__main__':
    main()