import os

from rafiki.utils.log import configure_logging
from rafiki.admin import Admin
from rafiki.admin.app import app

configure_logging('admin')

if __name__ == "__main__":
    # Run seed logic for admin at start-up
    admin = Admin()
    admin.seed()

    # Run Flask app
    app.run(
        host='0.0.0.0', 
        port=os.getenv('ADMIN_PORT', 3000), 
        threaded=True)
