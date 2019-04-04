import string
import random
import time

from rafiki.constants import TrainJobStatus

# Generates a random ID
def gen_id(length=16):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

# Blocks until a train job has stopped
def wait_until_train_job_has_stopped(client, app, timeout=60*20, tick=10):
    length = 0
    while True:
        train_job = client.get_train_job(app)
        status = train_job['status']

        if status == TrainJobStatus.ERRORED:
            raise Exception('Train job has errored.')
        elif status == TrainJobStatus.STOPPED:
            # Unblock
            return

        # Still running...
        if timeout is not None and length >= timeout:
            raise TimeoutError('Train job is running for too long')

        length += tick
        time.sleep(tick)