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
from rafiki.model import BaseModel, Params, KnobConfig, Knobs, \
                  DatasetUtils, CorpusDataset, ImageFilesDataset, \
                  LoggerUtils, \
                  utils, logger, dataset, load_model_class, parse_model_install_command, \
                  serialize_knob_config, deserialize_knob_config, \
                  BaseKnob, CategoricalKnob, IntegerKnob, FloatKnob, FixedKnob, ArchKnob, \
                  KnobValue, CategoricalValue, PolicyKnob

# rafiki/model/dataset.py
class TestDatasetUtils(object):
    pass

class TestModelDataset(object):
    pass

class TestCorpusDataset(object):
    pass

class TestImageFilesDataset(object):
    pass

class TestAudioFilesDataset(object):
    pass

def test__load_pil_images():
    pass


# rafiki/model/dev.py
def test_tune_model():
    pass

def test_make_predictions():
    pass

def test_test_model_class():
    pass

def test_warn_user():
    pass

def test_inform_user():
    pass

def test__pull_shared_params():
    pass

def test__evaluate_model():
    pass

def test__save_model():
    pass

def test__maybe_read_knobs_from_args():
    pass

def test__maybe_read_budget_from_args():
    pass

def test__check_model_class():
    pass

def test__check_dependencies():
    pass

def test__check_knob_config():
    pass

def test__assert_jsonable():
    pass

def test__check_model_inst():
    pass

def test__print_header():
    pass

# rafiki/model/knob.py
class TestKnobValue(object):
    pass

class TestCategoricalKnob(object):
    pass

class TestFixedKnob(object):
    pass

class TestPolicyKnob(object):
    pass

class TestIntegerKnob(object):
    pass

class TestFloatKnob(object):
    pass

class TestArchKnob(object):
    pass

# rafiki/model/log.py
class TestLoggerUtils(object):
    pass

class TestLoggerUtilsDebugHandler(object):
    pass

# rafiki/model/utils.py
def test_load_model_class():
    pass

def test_parse_model_install_command():
    pass

def test_parse_ctc_decoder_url():
    pass

def test_deserialize_knob_config():
    pass

def test_serialize_knob_config():
    pass

