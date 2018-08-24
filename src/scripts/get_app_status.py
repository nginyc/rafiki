import pprint

from admin import Admin

admin = Admin()

app_status = admin.get_app_status(
    name='fashion_mnist_app'
)

print('\n\n')
print('App Status:')
pprint.pprint(app_status)
