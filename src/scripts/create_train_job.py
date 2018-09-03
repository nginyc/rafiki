import sys

from admin import Admin
from common import BudgetType

if len(sys.argv) < 3:
    print('Usage: python {} <email> <password>'.format(__file__))
    exit(1)

email = sys.argv[1]
password = sys.argv[2]

admin = Admin()

user = admin.authenticate_user(email, password)

admin.create_train_job(
    user_id=user['id'],
    budget_type=BudgetType.TRIAL_COUNT,
    budget_amount=10,
    app_name='fashion_mnist_app',
    task='IMAGE_CLASSIFICATION_WITH_ARRAYS',
    train_dataset_uri='tf-keras://fashion_mnist?train_or_test=train',
    test_dataset_uri='tf-keras://fashion_mnist?train_or_test=test'
)
