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

import random
import numpy as np

from rafiki.model import BaseModel, IntegerKnob, FixedKnob, CategoricalKnob, FloatKnob, PolicyKnob

class Model(BaseModel):
    '''
    A mock model
    '''
    @staticmethod
    def get_knob_config():
        return {
            'int': IntegerKnob(1, 32),
            'float': FloatKnob(1e-5, 1),
            'cat': CategoricalKnob(['a', 'b', 'c']),
            'fixed': FixedKnob('fixed'),
            'policy': PolicyKnob('EARLY_STOP')
        }

    def train(self, dataset_path, **kwargs):
        pass

    def evaluate(self, dataset_path):
        return random.random()

    def predict(self, queries):
        return [1 for x in queries]

    def dump_parameters(self):
        return {'int': 100, 'str': 'str', 'float': 0.001, 'np': np.array([1, 2, 3])}

    def load_parameters(self, params):
        pass
