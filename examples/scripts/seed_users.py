import pprint
import os
import csv

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD

def seed_users(client, csv_file_path):
    with open(csv_file_path, 'rt', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lower() for name in reader.fieldnames]
        for row in reader:
            email = row['email']
            password = row['password']
            user_type = row['user_type']
            try:
                client.create_user(email, password, user_type)
            except Exception as e:
                print('Failed to create user `{}` due to:'.format(email))
                print(e)

if __name__ == '__main__':
    rafiki_host = os.environ.get('RAFIKI_HOST', 'localhost')
    admin_port = int(os.environ.get('ADMIN_EXT_PORT', 3000))
    admin_web_port = int(os.environ.get('ADMIN_WEB_EXT_PORT', 3001))
    user_email = os.environ.get('USER_EMAIL', SUPERADMIN_EMAIL)
    user_password = os.environ.get('USER_PASSWORD', SUPERADMIN_PASSWORD)
    csv_file_path = os.environ.get('CSV_FILE_PATH', 'examples/scripts/users.csv')

    # Initialize client
    client = Client(admin_host=rafiki_host, admin_port=admin_port)
    client.login(email=user_email, password=user_password)

    seed_users(client, csv_file_path)