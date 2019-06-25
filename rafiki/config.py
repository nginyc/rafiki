import os

# Global
APP_SECRET = os.environ.get('APP_SECRET', 'rafiki')
SUPERADMIN_EMAIL = 'superadmin@rafiki'
SUPERADMIN_PASSWORD = os.environ.get('SUPERADMIN_PASSWORD', 'rafiki')