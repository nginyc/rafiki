import numpy as np

from rafiki.constants import TaskType

def ensemble_predictions(predictions_list, task):
    if len(predictions_list) == 0 or len(predictions_list[0]) == 0:
        return []

    # By default, just return some trial's predictions

    mean_predictions = []
    for predictions in np.transpose(predictions_list, axes=[1, 0, 2]):
        mean_predictions.append(np.mean(predictions, axis=0))

    if task == TaskType.IMAGE_CLASSIFICATION:
        # Map probabilities to most probable label
        predictions = np.argmax(mean_predictions, axis=1)

    predictions = _simplify_predictions(predictions)

    return predictions

def _simplify_predictions(predictions):
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()

    return predictions