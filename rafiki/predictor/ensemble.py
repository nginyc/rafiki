import numpy as np

from rafiki.constants import TaskType

def ensemble_predictions(predictions_list, predict_label_mappings, task):
    # TODO: Better ensembling of predictions based on `predict_label_mapping` & `task` of models

    if len(predictions_list) == 0 or len(predictions_list[0]) == 0:
        return []

    # By default, just return some trial's predictions
    index = 0
    predictions = predictions_list[index]
    predict_label_mapping = predict_label_mappings[index]

    if task == TaskType.IMAGE_CLASSIFICATION:
        # Map probabilities to most probable label
        pred_indices = np.argmax(predictions, axis=1)
        predictions = [predict_label_mapping[str(i)] for i in pred_indices]
    
    return predictions
