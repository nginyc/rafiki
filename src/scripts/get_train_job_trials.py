import pprint
import sys

from admin import Admin

if len(sys.argv) < 3:
    print('Usage: python {} <app_name> <train_job_id>'.format(__file__))
    exit(1)

app_name = sys.argv[1]
train_job_id = sys.argv[2]

admin = Admin()

trials = admin.get_trials(
    app_name=app_name,
    train_job_id=train_job_id
)

print('\n\n')
print('Train Job\'s Trials:')
pprint.pprint(trials)
