import pytest
import json

from rafiki.model import IntegerKnob, CategoricalKnob, FloatKnob, FixedKnob, ListKnob, KnobValue, PolicyKnob
from rafiki.test.utils import global_setup
from rafiki.advisor.skopt import SkoptAdvisor
from rafiki.advisor.advisor import RandomAdvisor
from rafiki.advisor.tf import EnasAdvisor

class TestAdvisor():
    # Knob config for hyperparameter search
    @pytest.fixture(scope='class', autouse=True)
    def knob_config_hyp_search(self):
        return {
            'hidden_layer_units': IntegerKnob(2, 128),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
            'criterion': CategoricalKnob(['gini', 'entropy']),
            'image_size': FixedKnob(32),
            'quick_train': PolicyKnob('QUICK_TRAIN')
        }
    
    # Knob config for architecture search
    @pytest.fixture(scope='class', autouse=True)
    def knob_config_arch_search(self):
        op_knob_values = [KnobValue(x) for x in ['identity', 'conv1x1', 'conv3x3']]
        input_knob_values = [KnobValue(i) for i in range(2)]

        return {
            'arch': ListKnob([
                CategoricalKnob(input_knob_values), # Input 1
                CategoricalKnob(op_knob_values),  # Op on input 1
                CategoricalKnob(input_knob_values), # Input 2
                CategoricalKnob(op_knob_values) # Op on input 2
            ]),
            'image_size': FixedKnob(32),
            'quick_train': PolicyKnob('QUICK_TRAIN'), 
            'quick_eval': PolicyKnob('QUICK_EVAL')
        }

    # Knob config for hyperparameter + architecture search
    @pytest.fixture(scope='class', autouse=True)
    def knob_config_hyp_arch_search(self):
        op_knob_values = [KnobValue(x) for x in ['identity', 'conv1x1', 'conv3x3']]
        input_knob_values = [KnobValue(i) for i in range(2)]

        return {
            'arch': ListKnob([
                CategoricalKnob(input_knob_values), # Input 1
                CategoricalKnob(op_knob_values),  # Op on input 1
                CategoricalKnob(input_knob_values), # Input 2
                CategoricalKnob(op_knob_values) # Op on input 2
            ]),
            'hidden_layer_units': IntegerKnob(2, 128),
            'learning_rate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
            'criterion': CategoricalKnob(['gini', 'entropy']),
            'image_size': FixedKnob(32),
            'quick_train': PolicyKnob('QUICK_TRAIN'), 
            'quick_eval': PolicyKnob('QUICK_EVAL')
        }

    def test_skopt_advisor(self, knob_config_hyp_search):
        '''
        `SkoptAdvisor` should handle hyperparameter search
        '''
        advisor = SkoptAdvisor(knob_config_hyp_search)

        # Request first proposal
        proposal = advisor.propose(worker_id='localhost', trial_no=0, total_trials=2)
        assert proposal.is_valid
        self._check_knobs_for_hyp_search(proposal.knobs)
        advisor.feedback(1, proposal)

        # Request next proposal after feedback
        proposal = advisor.propose(worker_id='localhost', trial_no=1, total_trials=2)
        assert proposal.is_valid
        self._check_knobs_for_hyp_search(proposal.knobs)

    def test_random_advisor(self, knob_config_hyp_arch_search):
        '''
        `RandomAdvisor` should handle hyperparameter + architecture search
        '''
        advisor = RandomAdvisor(knob_config_hyp_arch_search)

        # Request first proposal
        proposal = advisor.propose(worker_id='localhost', trial_no=0, total_trials=2)
        assert proposal.is_valid
        self._check_knobs_for_hyp_arch_search(proposal.knobs)
        advisor.feedback(1, proposal)

        # Request next proposal after feedback
        proposal = advisor.propose(worker_id='localhost', trial_no=1, total_trials=2)
        assert proposal.is_valid
        self._check_knobs_for_hyp_arch_search(proposal.knobs)

    
    def test_enas_advisor(self, knob_config_arch_search):
        '''
        `EnasAdvisor` should handle architecture search
        '''
        advisor = EnasAdvisor(knob_config_arch_search)

        # Request first proposal
        proposal = advisor.propose(worker_id='localhost', trial_no=0, total_trials=2)
        assert proposal.is_valid
        self._check_knobs_for_arch_search(proposal.knobs)
        advisor.feedback(1, proposal)

        # Request next proposal after feedback
        proposal = advisor.propose(worker_id='localhost', trial_no=1, total_trials=2)
        assert proposal.is_valid
        self._check_knobs_for_arch_search(proposal.knobs)
    

    def _check_knobs_for_hyp_search(self, knobs):
        assert isinstance(knobs.get('hidden_layer_units'), int) and 2 <= knobs.get('hidden_layer_units') <= 128
        assert isinstance(knobs.get('learning_rate'), float) and 1e-5 <= knobs.get('learning_rate') <= 1e-1
        assert knobs.get('batch_size') in [16, 32, 64, 128]
        assert knobs.get('criterion') in ['gini', 'entropy']
        assert knobs.get('image_size') == 32
        assert knobs['quick_train'] in [True, False]

    def _check_knobs_for_arch_search(self, knobs):
        arch = knobs.get('arch')
        assert len(arch) == 4
        assert arch[0] in [0, 1]
        assert arch[1] in ['identity', 'conv1x1', 'conv3x3']
        assert arch[2] in [0, 1]
        assert arch[3] in ['identity', 'conv1x1', 'conv3x3']
        assert knobs.get('image_size') == 32
        assert knobs['quick_train'] in [True, False]
        assert knobs['quick_eval'] in [True, False]

    def _check_knobs_for_hyp_arch_search(self, knobs):
        arch = knobs.get('arch')
        assert len(arch) == 4
        assert arch[0] in [0, 1]
        assert arch[1] in ['identity', 'conv1x1', 'conv3x3']
        assert arch[2] in [0, 1]
        assert arch[3] in ['identity', 'conv1x1', 'conv3x3']
        assert isinstance(knobs.get('hidden_layer_units'), int) and 2 <= knobs.get('hidden_layer_units') <= 128
        assert isinstance(knobs.get('learning_rate'), float) and 1e-5 <= knobs.get('learning_rate') <= 1e-1
        assert knobs.get('batch_size') in [16, 32, 64, 128]
        assert knobs.get('criterion') in ['gini', 'entropy']
        assert knobs.get('image_size') == 32
        assert knobs['quick_train'] in [True, False]
        assert knobs['quick_eval'] in [True, False]
