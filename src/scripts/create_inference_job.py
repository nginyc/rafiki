import sys

from admin import Admin
from rafiki.constants import BudgetType

if len(sys.argv) < 3:
    print('Usage: python {} <email> <password>'.format(__file__))
    exit(1)

email = sys.argv[1]
password = sys.argv[2]

admin = Admin()

user = admin.authenticate_user(email, password)

admin.create_inference_job(
    user_id=user['id'],
    app='fashion_mnist_app'
)
