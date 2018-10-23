import numpy as np

from rafiki.constants import TaskType

def ensemble_predictions(predictions_list, train_dataset_meta, task):
    if len(predictions_list) == 0 or len(predictions_list[0]) == 0:
        return []

    # By default, just return some trial's predictions
    index = 0
    predictions = predictions_list[index]

    if task == TaskType.IMAGE_CLASSIFICATION:
        # Map probabilities to most probable label
        train_index_to_label = train_dataset_meta
        pred_indices = np.argmax(predictions, axis=1)
        predictions = [train_index_to_label[i] for i in pred_indices]
    
    return predictions
