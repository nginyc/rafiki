import os

from rafiki.utils.log import configure_logging
from rafiki.advisor import app

configure_logging('advisor')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv('ADVISOR_PORT', 8001), debug=True)
