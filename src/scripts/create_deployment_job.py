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

admin.create_deployment_job(
    user_id=user['id'],
    app_name='fashion_mnist_app'
)
