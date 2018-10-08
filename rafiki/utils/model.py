import os
import numpy as np
from importlib import import_module

TEMP_MODEL_FILE_NAME = 'temp'

def load_model_class(model_file_bytes, model_class):
    # Save the model file to disk
    f = open('{}.py'.format(TEMP_MODEL_FILE_NAME), 'wb')
    f.write(model_file_bytes)
    f.close()

    # Import model file as module
    mod = import_module(TEMP_MODEL_FILE_NAME)

    # Extract model class from module
    clazz = getattr(mod, model_class)

    # Remove temporary file
    os.remove(f.name)

    return clazz

def probabilities_to_predictions(probabilities):
    return np.argmax(probabilities, axis=1)

def parse_model_prediction(prediction):
    if isinstance(prediction, np.int64):
        return int(prediction)
    
    if isinstance(prediction, np.uint8):
        return int(prediction)

    return prediction