import os

from rafiki.utils.log import configure_logging
from rafiki.admin.app import app

configure_logging('admin')

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=os.getenv('ADMIN_PORT', 8000), 
        debug=True,
        threaded=True)
