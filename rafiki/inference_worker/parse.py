
import numpy as np

def to_json_serializable(data):
    if isinstance(data, np.int64):
        return int(data)

    return data