import os
import argparse

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.model import serialize_knob_config
from rafiki.constants import BudgetType, TaskType
from examples.models.image_classification.TfEnas import TfEnasBase

def run_enas(client, enable_gpu, full=True):
    client.create_model(
        name='TfEnasBase',
        task=TaskType.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/image_classification/TfEnas.py',
        model_class='TfEnasBase',
        dependencies={ 'tensorflow': '1.12.0' }
    )

    client.create_model(
        name='TfEnasSearch',
        task=TaskType.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/image_classification/TfEnas.py',
        model_class='TfEnasSearch',
        dependencies={ 'tensorflow': '1.12.0' }
    )

    knob_config = TfEnasBase.get_knob_config()
    knob_config_str = serialize_knob_config(knob_config)
    data = client.create_advisor(knob_config_str)
    advisor_id = data['id']

    client.create_train_job(
        app='cifar_10_enas_search',
        task=TaskType.IMAGE_CLASSIFICATION,
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
        budget={ 
            BudgetType.MODEL_TRIAL_COUNT: 31 * 10 if not full else 31 * 150,
            BudgetType.ENABLE_GPU: enable_gpu
        },
        models={
            'TfEnasSearch': {
                'advisor_id': advisor_id,
                'should_save': False,
                'knobs': { 'num_layers': 0, 'initial_block_ch': 4 } if not full else {}
            }
        }
    )

    client.create_train_job(
        app='cifar_10_enas_final',
        task=TaskType.IMAGE_CLASSIFICATION,
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
        budget={ 
            BudgetType.MODEL_TRIAL_COUNT: 3,
            BudgetType.ENABLE_GPU: enable_gpu
        },
        models={
            'TfEnasBase': {
                'advisor_id': advisor_id,
                'knobs': { 'trial_epochs': 10, 'initial_block_ch': 4 } if not full else {}
            }
        }
    )


if __name__ == '__main__':
    rafiki_host = os.environ.get('RAFIKI_HOST', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    admin_web_port = int(os.environ.get('ADMIN_WEB_EXT_PORT', 3001))
    user_email = os.environ.get('USER_EMAIL', SUPERADMIN_EMAIL)
    user_password = os.environ.get('USER_PASSWORD', SUPERADMIN_PASSWORD)
    enable_gpu = int(os.environ.get('ENABLE_GPU', 0))

    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client(admin_host=rafiki_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)
    admin_web_url = 'http://{}:{}'.format(rafiki_host, admin_web_port)

    # Run quickstart
    run_enas(client, enable_gpu, full=args.full)