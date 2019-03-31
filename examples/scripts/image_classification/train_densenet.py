import os
import argparse
import pprint

from rafiki.client import Client
from rafiki.model import serialize_knob_config
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import BudgetType, TaskType, ModelDependency

from examples.scripts.utils import gen_id, wait_until_train_job_has_stopped

def train_densenet(client, gpus, full):
    total_trials = 10 if not full else 100 

    app_id = gen_id()
    app = 'cifar_10_densenet_{}'.format(app_id)
    model_name = 'PyDenseNetBc_{}'.format(app_id)

    print('Creating model...')
    model = client.create_model(
        name=model_name,
        task=TaskType.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/image_classification/PyDenseNetBc.py',
        model_class='PyDenseNetBc',
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            ModelDependency.TORCHVISION: '0.2.2'
        }
    )
    pprint.pprint(model)
    
    print('Creating train job...')
    train_job = client.create_train_job(
        app=app,
        task=TaskType.IMAGE_CLASSIFICATION,
        train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
        val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
        budget={ 
            BudgetType.MODEL_TRIAL_COUNT: total_trials,
            BudgetType.GPU_COUNT: gpus
        },
        models={
            model_name: {
                'knobs': { 'batch_size': 32, 'max_trial_epochs': 10 } if not full else {}
            }
        }
    )
    pprint.pprint(train_job)
    wait_until_train_job_has_stopped(client, app)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Whether to do training for its full duration/capacity')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use')
    parser.add_argument('--host', type=str, default=os.environ.get('RAFIKI_ADDR'), help='Host of Rafiki instance')
    parser.add_argument('--admin_port', type=int, default=os.environ.get('ADMIN_EXT_PORT'), help='Port for Rafiki Admin on host')
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client(admin_host=args.host, admin_port=args.admin_port)
    client.login(email=args.email, password=args.password)

    train_densenet(client, args.gpus, args.full)