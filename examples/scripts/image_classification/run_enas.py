import os
import argparse
import pprint

from rafiki.client import Client
from rafiki.constants import BudgetType, TaskType, ModelDependency
from rafiki.advisor import AdvisorType
from rafiki.config import SUPERADMIN_EMAIL

from examples.models.image_classification.TfEnas import TfEnas
from examples.scripts.utils import gen_id, wait_until_train_job_has_stopped

def run_enas(client, gpus, train_strategy, num_cycles, enas_batch_size, num_eval_per_cycle, full=True):
    app_id = gen_id()
    num_cycles = 10 if not full else num_cycles
    period = num_eval_per_cycle + 1
    num_final_train_trials = 1
    trial_count = period * num_cycles + num_final_train_trials
    app = 'cifar_10_enas_{}'.format(app_id)
    model_name = 'TfEnas_{}'.format(app_id)

    print('Creating advisor...')
    knob_config = TfEnas.get_knob_config()
    advisor_config = { 'num_eval_per_cycle': num_eval_per_cycle, 
                        'batch_size': enas_batch_size, 
                        'train_strategy': train_strategy }
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
                    'knobs': { 'num_layers': 2, 'trial_epochs': 30, 
                                'early_stop_num_layers': 0 } if not full else {}
                }
            }
        )
        pprint.pprint(train_job)
        wait_until_train_job_has_stopped(client, app, timeout=None)

    finally:
        print('Stopping train job...')
        pprint.pprint(client.stop_train_job(app))

        print('Deleting advisor...')
        pprint.pprint(client.delete_advisor(advisor_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Whether to run ENAS in its full duration/capacity')
    parser.add_argument('--enas_batch_size', type=int, default=10, help='Batch size for ENAS controller')
    parser.add_argument('--train_strategy', type=str, default='ISOLATED', help='Train strategy for ENAS controller')
    parser.add_argument('--num_eval_per_cycle', type=int, default=300, help='No. of evaluation trials in a cycle of train-eval in ENAS')
    parser.add_argument('--num_cycles', type=int, default=150, help='No. of cycles of train-eval in ENAS')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use')
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
        
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)

    # Run ENAS
    run_enas(client, args.gpus, args.train_strategy, args.num_cycles, args.enas_batch_size, 
            args.num_eval_per_cycle, full=args.full)