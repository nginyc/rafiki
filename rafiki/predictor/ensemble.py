import numpy as np
from typing import List
from collections import Iterable

from rafiki.constants import TaskType

def ensemble_predictions(predictions: List[any], task: str) -> any:
    if len(predictions) == 0:
        return None

    # By default, just return some worker's predictions
    index = 0
    prediction = predictions[index]

    if task == TaskType.IMAGE_CLASSIFICATION:
        # All probs must have same length
        probs_by_worker = predictions
        assert all([len(x) == len(probs_by_worker[0]) for x in probs_by_worker]) 

        # Compute mean of probabilities across predictions
        probs = np.mean(probs_by_worker, axis=0)
        prediction = probs

    prediction = _simplify_prediction(prediction)

    return prediction

def _simplify_prediction(prediction):
    # Convert numpy arrays to lists
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    # Recurvely apply to elements of iterables
    if isinstance(prediction, Iterable):
        for (i, x) in enumerate(prediction):
            prediction[i] = _simplify_prediction(x)

    return prediction