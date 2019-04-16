import os

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL

if __name__ == '__main__':
    rafiki_host = os.environ.get('RAFIKI_HOST', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    user_email = SUPERADMIN_EMAIL
    user_password = os.environ.get('SUPERADMIN_PASSWORD', 'rafiki')

    # Initialize client
    client = Client(admin_host=rafiki_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)
    print(client.stop_all_jobs())