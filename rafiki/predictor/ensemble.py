#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import numpy as np
from collections import Iterable

from rafiki.constants import TaskType

def ensemble_predictions(predictions_list, task):
    if len(predictions_list) == 0 or len(predictions_list[0]) == 0:
        return []

    if task == TaskType.IMAGE_CLASSIFICATION:
        # Compute mean of probabilities across predictions 
        predictions = []
        for preds in np.transpose(predictions_list, axes=[1, 0, 2]):
            predictions.append(np.mean(preds, axis=0))
    elif task == TaskType.SPEECH_RECOGNITION:
        return predictions_list
    else:
        # By default, just return some trial's predictions
        index = 0
        predictions = predictions_list[index]

    predictions = _simplify_predictions(predictions)

    return predictions

def _simplify_predictions(predictions):
    # Convert numpy arrays to lists
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()

    if isinstance(predictions, Iterable):
        for i in range(len(predictions)):
            if isinstance(predictions[i], np.ndarray):
                predictions[i] = predictions[i].tolist()

    return predictions