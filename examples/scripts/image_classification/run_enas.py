import os
import argparse
import pprint

from rafiki.client import Client
from rafiki.model import serialize_knob_config
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import BudgetType, TaskType, ModelDependency
from examples.models.image_classification.TfEnas import TfEnasTrain

from examples.scripts.utils import gen_id, wait_until_train_job_has_stopped

def run_enas(client, enable_gpu, full=True):
    app_id = gen_id()
    search_app = 'cifar_10_enas_search_{}'.format(app_id)
    train_app = 'cifar_10_enas_train_{}'.format(app_id)
    search_model = 'TfEnasSearch_{}'.format(app_id)
    train_model = 'TfEnasTrain_{}'.format(app_id)
    search_trial_count = 31 * 10 if not full else 31 * 150
    train_trial_count = 3

    print('Creating advisor...')
    knob_config = TfEnasTrain.get_knob_config()
    knob_config_str = serialize_knob_config(knob_config)
    advisor = client.create_advisor(knob_config_str)
    pprint.pprint(advisor)
    advisor_id = advisor['id']

    try:
        print('Creating model for ENAS search...')
        model = client.create_model(
            name=search_model,
            task=TaskType.IMAGE_CLASSIFICATION,
            model_file_path='examples/models/image_classification/TfEnas.py',
            model_class='TfEnasSearch',
            dependencies={ ModelDependency.TENSORFLOW: '1.12.0' }
        )
        pprint.pprint(model)
        
        print('Creating train job for ENAS search...')
        train_job = client.create_train_job(
            app=search_app,
            task=TaskType.IMAGE_CLASSIFICATION,
            train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
            val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
            budget={ 
                BudgetType.MODEL_TRIAL_COUNT: search_trial_count,
                BudgetType.ENABLE_GPU: 1 if enable_gpu else 0
            },
            models={
                search_model: {
                    'advisor_id': advisor_id,
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
            dependencies={ ModelDependency.TENSORFLOW: '1.12.0' }
        )
        pprint.pprint(model)

        print('Creating train job for training final models sampled from ENAS...')
        train_job = client.create_train_job(
            app=train_app,
            task=TaskType.IMAGE_CLASSIFICATION,
            train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
            val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
            budget={ 
                BudgetType.MODEL_TRIAL_COUNT: train_trial_count,
                BudgetType.ENABLE_GPU: enable_gpu
            },
            models={
                train_model: {
                    'advisor_id': advisor_id,
                    'knobs': { 'num_layers': 8, 'trial_epochs': 10, 'initial_block_ch': 4 } if not full else {}
                }
            }
        )
        pprint.pprint(train_job)
        wait_until_train_job_has_stopped(client, train_app)

    finally:
        print('Deleting advisor...')
        client.delete_advisor(advisor_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Whether to run ENAS in its full duration/capacity')
    parser.add_argument('--enable_gpu', action='store_true', help='Whether to use GPU')
    parser.add_argument('--host', type=str, default=os.environ.get('RAFIKI_ADDR'), help='Host of Rafiki instance')
    parser.add_argument('--admin_port', type=int, default=os.environ.get('ADMIN_EXT_PORT'), help='Port for Rafiki Admin on host')
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client(admin_host=args.host, admin_port=args.admin_port)
    client.login(email=args.email, password=args.password)

    # Run ENAS
    run_enas(client, args.enable_gpu, full=args.full)