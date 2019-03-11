import os
import argparse
import pprint

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.model import serialize_knob_config
from rafiki.constants import BudgetType, TaskType
from examples.models.image_classification.TfEnas import TfEnasTrain

from examples.scripts.utils import gen_id, wait_until_train_job_has_stopped

def run_enas(client, enable_gpu, full=True):
    app_id = gen_id()
    search_app = 'cifar_10_enas_search_{}'.format(app_id)
    train_app = 'cifar_10_enas_train_{}'.format(app_id)
    search_model = 'TfEnasSearch_{}'.format(app_id)
    train_model = 'TfEnasTrain_{}'.format(app_id)

    print('Creating advisor...')
    knob_config = TfEnasTrain.get_knob_config()
    knob_config_str = serialize_knob_config(knob_config)
    advisor = client.create_advisor(knob_config_str)
    pprint.pprint(advisor)
    advisor_id = advisor['id']

    print('Creating model for ENAS search...')
    model = client.create_model(
        name=search_model,
        task=TaskType.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/image_classification/TfEnas.py',
        model_class='TfEnasSearch',
        dependencies={ 'tensorflow': '1.12.0' }
    )
    pprint.pprint(model)
    
    print('Creating train job for ENAS search...')
    train_job = client.create_train_job(
        app=search_app,
        task=TaskType.IMAGE_CLASSIFICATION,
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
        budget={ 
            BudgetType.MODEL_TRIAL_COUNT: 31 * 10 if not full else 31 * 150,
            BudgetType.ENABLE_GPU: enable_gpu
        },
        models={
            search_model: {
                'advisor_id': advisor_id,
                'should_save': False,
                'knobs': { 'num_layers': 0, 'initial_block_ch': 4 } if not full else {}
            }
        }
    )
    pprint.pprint(train_job)
    wait_until_train_job_has_stopped(client, search_app)

    print('Creating model for training final models sampled from ENAS...')
    model = client.create_model(
        name=train_model,
        task=TaskType.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/image_classification/TfEnas.py',
        model_class='TfEnasTrain',
        dependencies={ 'tensorflow': '1.12.0' }
    )
    pprint.pprint(model)

    print('Creating train job for training final models sampled from ENAS...')
    train_job = client.create_train_job(
        app=train_app,
        task=TaskType.IMAGE_CLASSIFICATION,
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
        budget={ 
            BudgetType.MODEL_TRIAL_COUNT: 3,
            BudgetType.ENABLE_GPU: enable_gpu
        },
        models={
            train_model: {
                'advisor_id': advisor_id,
                'knobs': { 'num_layers': 4, 'trial_epochs': 10, 'initial_block_ch': 4 } if not full else {}
            }
        }
    )
    pprint.pprint(train_job)
    wait_until_train_job_has_stopped(client, train_app)

if __name__ == '__main__':
    rafiki_host = os.environ.get('RAFIKI_HOST', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    admin_web_port = int(os.environ.get('ADMIN_WEB_EXT_PORT', 3001))
    user_email = os.environ.get('USER_EMAIL', SUPERADMIN_EMAIL)
    user_password = os.environ.get('USER_PASSWORD', SUPERADMIN_PASSWORD)
    enable_gpu = int(os.environ.get('ENABLE_GPU', 0))

    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Whether to run ENAS in its full duration/capacity')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client(admin_host=rafiki_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)
    admin_web_url = 'http://{}:{}'.format(rafiki_host, admin_web_port)

    # Run ENAS
    run_enas(client, enable_gpu, full=args.full)