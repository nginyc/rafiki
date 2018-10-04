import os

from rafiki.utils.log import configure_logging
from rafiki.advisor.app import app

configure_logging('advisor')

if __name__ == "__main__":
    # No threading since data is in-memory
    app.run(host='0.0.0.0', port=os.getenv('ADVISOR_PORT', 8001), threaded=False, debug=True)
