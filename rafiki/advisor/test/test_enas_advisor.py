import pytest
import json
import random

from rafiki.test.utils import global_setup
from rafiki.advisor import ListKnob, CategoricalKnob, FixedKnob
from rafiki.advisor.types.enas_advisor import EnasAdvisor

class TestEnasAdvisor():
    @pytest.fixture(scope='class', autouse=True)
    def knob_config(self):
        '''
        Initializes the flask app and returns the client for it
        '''
        return {
            'cell_arch': ListKnob(4, items=[
                CategoricalKnob([0, 1]), # Index 1
                CategoricalKnob([0, 1]), # Index 2
                CategoricalKnob(['identity', 'conv1x1', 'conv3x3']), # Op 1
                CategoricalKnob(['identity', 'conv1x1', 'conv3x3']), # Op 2
            ]),
            'learning_rate': FixedKnob(0.001)
        }

    def test_basic(self, knob_config):
        '''
        Basic operations should work
        '''
        advisor = EnasAdvisor()
        advisor.start(knob_config)

        # Request first set of knobs
        (knobs, _) = advisor.propose()
        self._check_knobs(knobs)

        advisor.feedback(1, knobs)

        # Request next set of knobs after feedback
        (knobs, _) = advisor.propose()
        self._check_knobs(knobs)

    def test_learning_best_knobs(self, knob_config):
        '''
        If there is a definitive best set of knobs, advisor should propose it eventually 
        '''
        advisor = EnasAdvisor()
        advisor.start(knob_config)
        best_knobs = {
            'cell_arch': [0, 1, 'conv3x3', 'conv3x3'],
            'learning_rate': 0.001
        }

        # Periodically give feedback that this knobs are the best
        for i in range(100):
            (knobs, _) = advisor.propose()
            if i % 2 == 0 and knobs != best_knobs:
                # Tell advisor it's bad
                advisor.feedback(random.random() * 0.1, knobs)
            else:
                # Tell advisor it's the best
                advisor.feedback(1, best_knobs)
        
        # Advisor should propose it
        proposed_knobs = [advisor.propose()[0] for _ in range(3)]
        assert best_knobs in proposed_knobs

    def test_learning_list_sequence(self, knob_config):
        '''
        Learn that these sequences for `cell_arch` are best:
        1 -> X -> 'conv1x1' -> X
        X -> 0 -> X -> 'conv3x3' 
        '''
        advisor = EnasAdvisor()
        advisor.start(knob_config)
        best_knobs = {
            'cell_arch': [1, 0, 'conv1x1', 'conv3x3'],
            'learning_rate': 0.001
        }

        # Periodically give feedback that this knobs are the best
        for _ in range(100):
            (knobs, _) = advisor.propose()
            cell_arch = knobs['cell_arch']
            if (cell_arch[0] == 1 and cell_arch[2] == 'conv1x1') or (cell_arch[1] == 0 and cell_arch[3] == 'conv3x3'):
                # Tell advisor it's good
                advisor.feedback(0.9 + random.random() * 0.1, best_knobs)
            else:
                # Tell advisor it's bad
                advisor.feedback(random.random() * 0.1, knobs)
        
        # Advisor should propose it
        proposed_knobs = [advisor.propose()[0] for _ in range(3)]
        assert best_knobs in proposed_knobs
        
    def _check_knobs(self, knobs):
        assert len(knobs) == 2

        assert 'cell_arch' in knobs
        cell_arch = knobs['cell_arch']
        assert len(cell_arch) == 4
        assert cell_arch[0] in [0, 1]
        assert cell_arch[1] in [0, 1]
        assert cell_arch[2] in ['identity', 'conv1x1', 'conv3x3']
        assert cell_arch[3] in ['identity', 'conv1x1', 'conv3x3']

        assert 'learning_rate' in knobs
        assert knobs['learning_rate'] == 0.001

        