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

import logging
import numpy as np
from typing import List, Callable, Any

logger = logging.getLogger(__name__)

def get_ensemble_method(task: str) -> Callable[[List[Any]], Any]:
    if task == 'IMAGE_CLASSIFICATION':
        return ensemble_probabilities
    else:
        return ensemble

def ensemble_probabilities(predictions: List[Any]) -> Any:
    if len(predictions) == 0:
        return None

    # All probs must have same length
    probs_by_worker = predictions
    assert all([len(x) == len(probs_by_worker[0]) for x in probs_by_worker])

    # Compute mean of probabilities across predictions
    probs = np.mean(probs_by_worker, axis=0)
    prediction = probs
    prediction = _simplify_prediction(prediction)
    return prediction

def ensemble(predictions: List[Any]) -> Any:
    if len(predictions) == 0:
        return None

    # Return some worker's predictions
    index = 0
    prediction = predictions[index]
    prediction = _simplify_prediction(prediction)
    return prediction

def _simplify_prediction(prediction):
    # Convert numpy arrays to lists
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    # Recurvely apply to elements of lists
    if isinstance(prediction, list):
        for (i, x) in enumerate(prediction):
            prediction[i] = _simplify_prediction(x)

    return prediction
