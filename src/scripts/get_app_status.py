import pprint

from admin import Admin
from common import TfKerasDatasetConfig

admin = Admin()

app_status = admin.get_app_status(
    name='fashion_mnist_app'
)

pprint.pprint(app_status)
