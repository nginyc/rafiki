from inference_worker import InferenceWorker
import signal
import sys

def sigterm_handler(_signo, _stack_frame):
    print("sigterm_handler executed, %s, %s" % (_signo, _stack_frame))
    sys.exit(0)

if __name__ == '__main__':
    inference_worker = InferenceWorker()
    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        inference_worker.start()
    finally:
        inference_worker.stop()
        print('Inference worker stopped gracefully.')
    