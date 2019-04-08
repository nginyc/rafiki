import os
import argparse
import pprint

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import BudgetType, TaskType, ModelDependency
from rafiki.advisor import AdvisorType

from examples.models.image_classification.TfEnas import TfEnas
from examples.scripts.utils import gen_id, wait_until_train_job_has_stopped

def run_enas(client, gpus, full=True):
    app_id = gen_id()
    num_eval_trials = 60 if not full else 300
    batch_size = 2 if not full else 10
    period = num_eval_trials + 1
    num_final_train_trials = 1
    trial_count = period * 10 + num_final_train_trials if not full else period * 150 + num_final_train_trials
    app = 'cifar_10_enas_{}'.format(app_id)
    model_name = 'TfEnas_{}'.format(app_id)

    print('Creating advisor...')
    knob_config = TfEnas.get_knob_config()
    advisor_config = { 'num_eval_trials': num_eval_trials, 'batch_size': batch_size }
    advisor = client.create_advisor(knob_config, advisor_type=AdvisorType.ENAS, advisor_config=advisor_config)
    pprint.pprint(advisor)
    advisor_id = advisor['id']

    try:
        print('Creating model...')
        model = client.create_model(
            name=model_name,
            task=TaskType.IMAGE_CLASSIFICATION,
            model_file_path='examples/models/image_classification/TfEnas.py',
            model_class='TfEnas',
            dependencies={ ModelDependency.TENSORFLOW: '1.12.0' }
        )
        pprint.pprint(model)
        
        print('Creating train job...')
        train_job = client.create_train_job(
            app=app,
            task=TaskType.IMAGE_CLASSIFICATION,
            train_dataset_uri='data/cifar_10_for_image_classification_train.zip',
            val_dataset_uri='data/cifar_10_for_image_classification_val.zip',
            budget={ 
                BudgetType.MODEL_TRIAL_COUNT: trial_count,
                BudgetType.GPU_COUNT: gpus
            },
            models={
                model_name: {
                    'advisor_id': advisor_id,
                    'knobs': { 'num_layers': 2, 'early_stop_num_layers': 0 } if not full else {}
                }
            }
        )
        pprint.pprint(train_job)
        wait_until_train_job_has_stopped(client, app, timeout=None)

    finally:
        print('Deleting advisor...')
        client.delete_advisor(advisor_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Whether to run ENAS in its full duration/capacity')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use')
    parser.add_argument('--host', type=str, default=os.environ.get('RAFIKI_ADDR'), help='Host of Rafiki instance')
    parser.add_argument('--admin_port', type=int, default=os.environ.get('ADMIN_EXT_PORT'), help='Port for Rafiki Admin on host')
    parser.add_argument('--advisor_port', type=int, default=os.environ.get('ADVISOR_EXT_PORT'), help='Port for Rafiki Advisor on host')
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client(admin_host=args.host, admin_port=args.admin_port, advisor_host=args.host, advisor_port=args.advisor_port)
    client.login(email=args.email, password=args.password)

    # Run ENAS
    run_enas(client, args.gpus, full=args.full)