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

import abc
import numpy as np

from rafiki.constants import AdvisorType

class InvalidAdvisorTypeException(Exception): pass

class BaseAdvisor(abc.ABC):
    '''
    Rafiki's base advisor class
    '''   

    @abc.abstractmethod
    def __init__(self, knob_config):
        raise NotImplementedError()

    @abc.abstractmethod
    def propose(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, knobs, score):
        raise NotImplementedError()

# Generalized Advisor class that wraps & hides implementation-specific advisor class
class Advisor():
    def __init__(self, knob_config, advisor_type=AdvisorType.BTB_GP):
        self._advisor = self._make_advisor(knob_config, advisor_type)
        self._knob_config = knob_config

    @property
    def knob_config(self):
        return self._knob_config

    def propose(self):
        knobs = self._advisor.propose()

        # Simplify knobs to use JSON serializable values
        knobs = {
            name: self._simplify_value(value)
                for name, value
                in knobs.items()
        }

        return knobs

    def feedback(self, knobs, score):
        self._advisor.feedback(knobs, score)

    def _make_advisor(self, knob_config, advisor_type):
        if advisor_type == AdvisorType.BTB_GP:
            from .btb_gp_advisor import BtbGpAdvisor
            return BtbGpAdvisor(knob_config)
        else:
            raise InvalidAdvisorTypeException()

    def _simplify_value(self, value):
        # TODO: Support int64 & other non-serializable data formats
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)

        return value
