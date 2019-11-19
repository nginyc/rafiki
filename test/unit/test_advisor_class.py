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
from rafiki.advisor import BaseAdvisor, make_advisor

# rafiki/advisor/advisor.py
def test_make_advisor():
    pass

def test__get_advisor_class_from_type():
    pass

class TestBaseAdvisor(object):
    pass

class TestFixedAdvisor(object):
    pass

class TestRandomAdvisor(object):
    pass

# rafiki/advisor/skopt.py
class TestBayesOptAdvisor(object):
    pass

class TestBayesOptWithParamSharingAdvisor(object):
    pass

def test__propose_exp_greedy_param():
    pass

def test__knob_to_dimension():
    pass

def test__simplify_value():
    pass

# rafiki/advisor/tf.py
class TestEnasAdvisor(object):
    pass

class TestEnasArchAdvisor(object):
    pass

