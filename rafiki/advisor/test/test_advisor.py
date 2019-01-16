import pytest
import json

from rafiki.model import IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob
from rafiki.test.utils import global_setup
from rafiki.advisor.advisor import Advisor, AdvisorType

class TestAdvisor():
    @pytest.fixture(scope='class', autouse=True)
    def knob_config(self):
        '''
        Initializes the flask app and returns the client for it
        '''
        return {
            'hidden_layer_units': IntegerKnob(2, 128),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
            'criterion': CategoricalKnob(['gini', 'entropy']),
            'image_size': FixedKnob(32)
        }

    def test_skopt_advisor(self, knob_config):
        '''
        Advisor of `SkoptAdvisor` type should work
        '''
        advisor = Advisor(knob_config, advisor_type=AdvisorType.SKOPT)

        # Request first set of knobs
        knobs = advisor.propose()
        self._check_knobs(knobs)
        advisor.feedback(knobs, 1)

        # Request next set of knobs after feeback
        knobs = advisor.propose()
        self._check_knobs(knobs)

    def test_btb_gp_advisor(self, knob_config):
        '''
        Advisor of `BtbGpAdvisor` type should work
        '''
        advisor = Advisor(knob_config, advisor_type=AdvisorType.BTB_GP)

        # Request first set of knobs
        knobs = advisor.propose()
        self._check_knobs(knobs)
        advisor.feedback(knobs, 1)

        # Request next set of knobs after feeback
        knobs = advisor.propose()
        self._check_knobs(knobs)
        
    def _check_knobs(self, knobs):
        '''
        Logins as superadmin and returns required auth headers
        '''
        assert isinstance(knobs.get('hidden_layer_units'), int) and 2 <= knobs.get('hidden_layer_units') <= 128
        assert isinstance(knobs.get('learning_rate'), float) and 1e-5 <= knobs.get('learning_rate') <= 1e-1
        assert knobs.get('batch_size') in [16, 32, 64, 128]
        assert knobs.get('criterion') in ['gini', 'entropy']
        assert knobs.get('image_size') == 32





