import pprint
import time
import requests

from rafiki.client import Client
from rafiki.constants import TaskType, BudgetType

ADMIN_HOST = 'localhost'
ADMIN_PORT = 8000
USER_EMAIL = 'superadmin@rafiki'
USER_PASSWORD = 'rafiki'
APP = 'fashion_mnist_app'
TRAIN_DATASET_URI = 'https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_train.zip?raw=true'
TEST_DATASET_URI = 'https://github.com/cadmusthefounder/mnist_data/blob/master/output/fashion_test.zip?raw=true'
QUERY = \
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], 
    [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], 
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], 
    [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], 
    [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], 
    [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], 
    [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], 
    [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], 
    [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], 
    [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], 
    [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def create_models(client):
    # Add TfSingleHiddenLayer model to Rafiki
    pprint.pprint(
        client.create_model(
            name='TfSingleHiddenLayer',
            task=TaskType.IMAGE_CLASSIFICATION,
            model_file_path='examples/models/image_classification/TfSingleHiddenLayer.py',
            model_class='TfSingleHiddenLayer'
        )
    )

    # Add SkDt model to Rafiki
    pprint.pprint(
        client.create_model(
            name='SkDt',
            task=TaskType.IMAGE_CLASSIFICATION,
            model_file_path='examples/models/image_classification/SkDt.py',
            model_class='SkDt'
        )
    )

def create_train_job(client):
    pprint.pprint(
        client.create_train_job(
            app=APP,
            task=TaskType.IMAGE_CLASSIFICATION,
            train_dataset_uri=TRAIN_DATASET_URI,
            test_dataset_uri=TEST_DATASET_URI,
            budget_type=BudgetType.MODEL_TRIAL_COUNT,
            budget_amount=2
        )
    )

def wait_until_train_job_has_completed(client):
    while True:
        time.sleep(10)
        try:
            train_job = client.get_train_job(app=APP)
            if train_job.get('status') == 'COMPLETED':
                return

        except:
            pass

def list_best_trials_of_train_job(client):
    pprint.pprint(
        client.get_best_trials_of_train_job(app=APP)
    )

def create_inference_job(client):
    pprint.pprint(
        client.create_inference_job(app=APP)
    )

# Returns `predictor_host` of inference job
def wait_until_inference_job_is_running(client):
    while True:
        # Give inference job deployment a bit of time
        time.sleep(20)
        try:
            inference_job = client.get_running_inference_job(app=APP)
            if inference_job.get('status') == 'RUNNING':
                return inference_job.get('predictor_host')

        except:
            pass

def make_predictions(client, predictor_host):
    res = requests.post(
        url='http://{}/predict'.format(predictor_host),
        json={ 'query': QUERY }
    )

    if res.status_code != 200:
        raise Exception(res.text)

    pprint.pprint(res.json())

def stop_inference_job(client):
    pprint.pprint(client.stop_inference_job(app=APP))

if __name__ == '__main__':
    client = Client(admin_host=ADMIN_HOST, admin_port=ADMIN_PORT)
    client.login(email=USER_EMAIL, password=USER_PASSWORD)

    print('Adding models to Rafiki...') 
    create_models(client)

    print('Creating train job for app "{}" on Rafiki...'.format(APP)) 
    create_train_job(client)

    print('Waiting for train job to complete...')
    wait_until_train_job_has_completed(client)
    print('Train job has been completed!')

    print('Listing best trials of latest train job for app "{}"...'.format(APP))
    list_best_trials_of_train_job(client)

    print('Creating inference job for app "{}" on Rafiki...'.format(APP))
    create_inference_job(client)

    print('Waiting for inference job to be running...')
    predictor_host = wait_until_inference_job_is_running(client)
    print('Inference job is running!')

    print('Making predictions...')
    make_predictions(client, predictor_host)

    print('Stopping inference job...')
    stop_inference_job(client)
