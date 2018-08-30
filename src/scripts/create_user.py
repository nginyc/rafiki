import sys

from admin import Admin

if len(sys.argv) < 4:
    print('Usage: python {} <email> <password> <user_type>'.format(__file__))
    exit(1)

admin = Admin()

email = sys.argv[1]
password = sys.argv[2]
user_type = sys.argv[3]

admin.create_user(
    email=email,
    password=password,
    user_type=user_type
)
