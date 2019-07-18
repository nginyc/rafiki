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

import pytest

from rafiki.model import IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, ArchKnob, KnobValue, PolicyKnob
from rafiki.advisor import make_advisor
from rafiki.advisor.advisor import FixedAdvisor, RandomAdvisor
from rafiki.advisor.skopt import BayesOptAdvisor, BayesOptWithParamSharingAdvisor
from rafiki.advisor.tf import EnasAdvisor
from rafiki.constants import BudgetOption

from test.utils import global_setup

class TestMakeAdvisor():
    @pytest.fixture(scope='class')
    def budget(self):
        return {
            BudgetOption.TIME_HOURS: 1
        }

    @pytest.fixture(scope='class')
    def day_budget(self):
        return {
            BudgetOption.TIME_HOURS: 24
        }

    @pytest.fixture(scope='class')
    def arch_knob(self):
        op_knob_values = [KnobValue(x) for x in ['identity', 'conv1x1', 'conv3x3']]
        input_knob_values = [KnobValue(i) for i in range(2)]
        return ArchKnob([
            input_knob_values, # Input 1
            op_knob_values,  # Op on input 1
            input_knob_values, # Input 2
            op_knob_values # Op on input 2
        ])

    # Knob config with only fixed & policy knobs is handled by `FixedAdvisor`
    def test_fixed_knobs(self, budget):
        knob_config = {
            'fixed': FixedKnob('fixed'),
            'fixed2': FixedKnob(2),
            'early_stop': PolicyKnob('EARLY_STOP')
        }
        advisor = make_advisor(knob_config, budget)
        assert isinstance(advisor, FixedAdvisor)

    # Knob config with only float, integer, categorical knobs is handled by `BayesOptAdvisor`
    def test_standard_knobs(self, budget):
        knob_config = {
            'int': IntegerKnob(2, 128),
            'float': FloatKnob(1e-5, 1e-1, is_exp=True),
            'cat': CategoricalKnob([16, 32, 64, 128]),
        }
        advisor = make_advisor(knob_config, budget)
        assert isinstance(advisor, BayesOptAdvisor)

    # Knob config with only float, integer, categorical knobs and `EARLY_STOP` policy is handled by `BayesOptAdvisor`
    def test_standard_knobs_with_early_stop(self, budget):
        knob_config = {
            'int': IntegerKnob(2, 128),
            'float': FloatKnob(1e-5, 1e-1, is_exp=True),
            'cat': CategoricalKnob([16, 32, 64, 128]),
            'early_stop': PolicyKnob('EARLY_STOP')
        }
        advisor = make_advisor(knob_config, budget)
        assert isinstance(advisor, BayesOptAdvisor)

    # Knob config with only float, integer, categorical knobs and `SHARE_PARAMS` policy is handled by `BayesOptWithParamSharingAdvisor`
    def test_standard_knobs_with_params_sharing(self, budget):
        knob_config = {
            'int': IntegerKnob(2, 128),
            'float': FloatKnob(1e-5, 1e-1, is_exp=True),
            'cat': CategoricalKnob([16, 32, 64, 128]),
            'share_params': PolicyKnob('SHARE_PARAMS')
        }
        advisor = make_advisor(knob_config, budget)
        assert isinstance(advisor, BayesOptWithParamSharingAdvisor)
    
    # Knob config with just architecture knobs is handled by `RandomAdvisor`
    def test_arch_knobs(self, budget, arch_knob):
        knob_config = {
            'arch': arch_knob,
            'arch2': arch_knob
        }
        advisor = make_advisor(knob_config, budget)
        assert isinstance(advisor, RandomAdvisor)

    # Knob config with architecture, float & categorical knobs is handled by `RandomAdvisor`
    def test_arch_knobs_with_standard_knobs(self, budget, arch_knob):
        knob_config = {
            'arch': arch_knob,
            'float': FloatKnob(1e-5, 1e-1, is_exp=True),
            'cat': CategoricalKnob([16, 32, 64, 128]),
        }
        advisor = make_advisor(knob_config, budget)
        assert isinstance(advisor, RandomAdvisor)

    # Knob config with an architecture knob, fixed knobs, and specific policies `SHARE_PARAMS`, `DOWNSCALE`, 
    # `EARLY_STOP`, `SKIP_TRAIN`, `QUICK_EVAL`, with day budget is handled by `EnasAdvisor`
    def test_arch_knob_with_enas_policies(self, day_budget, arch_knob):
        knob_config = {
            'arch': arch_knob,
            'fixed': FixedKnob('fixed'),
            'share_params': PolicyKnob('SHARE_PARAMS'),
            'downscale': PolicyKnob('DOWNSCALE'),
            'early_stop': PolicyKnob('EARLY_STOP'),
            'skip_train': PolicyKnob('SKIP_TRAIN'),
            'quick_eval': PolicyKnob('QUICK_EVAL')
        }
        advisor = make_advisor(knob_config, day_budget)
        assert isinstance(advisor, EnasAdvisor)

    # Knob config with an architecture knob but missing some ENAS policies is handled by `RandomAdvisor`
    def test_arch_knob_without_enas_policies(self, day_budget, arch_knob):
        knob_config = {
            'arch': arch_knob,
            'downscale': PolicyKnob('DOWNSCALE'),
            'early_stop': PolicyKnob('EARLY_STOP'),
            'skip_train': PolicyKnob('SKIP_TRAIN')
            # No quick eval
        }
        advisor = make_advisor(knob_config, day_budget)
        assert isinstance(advisor, RandomAdvisor)

    # Knob config with an architecture knob but insufficient budget is handled by `RandomAdvisor`
    def test_arch_knob_without_budget(self, budget, arch_knob):
        knob_config = {
            'arch': arch_knob,
            'fixed': FixedKnob('fixed'),
            'downscale': PolicyKnob('DOWNSCALE'),
            'early_stop': PolicyKnob('EARLY_STOP'),
            'skip_train': PolicyKnob('SKIP_TRAIN'),
            'quick_eval': PolicyKnob('QUICK_EVAL')
        }
        advisor = make_advisor(knob_config, budget)
        assert isinstance(advisor, RandomAdvisor)
