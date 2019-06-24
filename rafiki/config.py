import os

# Global
APP_SECRET = os.environ.get('APP_SECRET', 'rafiki')
SUPERADMIN_EMAIL = 'superadmin@rafiki'
SUPERADMIN_PASSWORD = os.environ.get('SUPERADMIN_PASSWORD', 'rafiki')

# Predictor
PREDICTOR_PREDICT_SLEEP = 0.25

# Inference worker
INFERENCE_WORKER_SLEEP = 0.25
INFERENCE_WORKER_PREDICT_BATCH_SIZE = 32