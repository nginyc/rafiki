import pprint
import sys

from admin import Admin

if len(sys.argv) < 2:
    print('Usage: python {} <train_job_id>'.format(__file__))
    exit(1)

train_job_id = sys.argv[1]

admin = Admin()

trials = admin.get_trials(
    app_name='fashion_mnist_app',
    train_job_id=train_job_id
)

print('\n\n')
print('Train Job\'s Trials:')
pprint.pprint(trials)
