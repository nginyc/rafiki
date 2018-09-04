import sys

from worker import Worker

if len(sys.argv) < 2:
    print('Usage: python {} <worker_id>'.format(__file__))
    exit(1)

worker_id = sys.argv[1]

worker = Worker(worker_id)
worker.start()