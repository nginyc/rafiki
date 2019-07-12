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

from btb.tuning import GP
from btb import HyperParameter, ParamTypes

from rafiki.model import BaseKnob, FloatKnob, IntegerKnob, CategoricalKnob, FixedKnob
from .advisor import BaseAdvisor

class BtbGpAdvisor(BaseAdvisor):
    '''
    Uses BTB's GP tuner
    '''   
    def __init__(self, knob_config):
        # TODO: Support conditional knobs
        tunables = self._get_tunables(knob_config)

        # TODO: Allow configuration of tuner
        self._tuner = GP(tunables=tunables)

    def propose(self):
        knobs = self._tuner.propose()
        return knobs

    def feedback(self, knobs, score):
        self._tuner.add(knobs, score)

    def _get_tunables(self, knob_config):
        tunables = [
            (name, _knob_to_tunable(x))
                for (name, x)
                in knob_config.items()
        ]
        return tunables

def _knob_to_tunable(knob):
    if isinstance(knob, CategoricalKnob):
        if knob.value_type is int:
            return HyperParameter(ParamTypes.INT_CAT, knob.values)
        elif knob.value_type is float:
            return HyperParameter(ParamTypes.FLOAT_CAT, knob.values)
        elif knob.value_type is str:
            return HyperParameter(ParamTypes.STRING, knob.values)
        elif knob.value_type is bool:
            return HyperParameter(ParamTypes.BOOL, knob.values)
    elif isinstance(knob, FixedKnob):
        if knob.value_type is int:
            return HyperParameter(ParamTypes.INT_CAT, [knob.value])
        elif knob.value_type is float:
            return HyperParameter(ParamTypes.FLOAT_CAT, [knob.value])
        elif knob.value_type is str:
            return HyperParameter(ParamTypes.STRING, [knob.value])
        elif knob.value_type is bool:
            return HyperParameter(ParamTypes.BOOL, [knob.value])
    elif isinstance(knob, IntegerKnob):
        if knob.is_exp:
            return HyperParameter(ParamTypes.INT_EXP, [knob.value_min, knob.value_max])
        else:
            return HyperParameter(ParamTypes.INT, [knob.value_min, knob.value_max])
    elif isinstance(knob, FloatKnob):
        if knob.is_exp:
            return HyperParameter(ParamTypes.FLOAT_EXP, [knob.value_min, knob.value_max])
        else:
            return HyperParameter(ParamTypes.FLOAT, [knob.value_min, knob.value_max])
    