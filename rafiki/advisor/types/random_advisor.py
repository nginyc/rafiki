import random
from .. import BaseAdvisor, UnsupportedKnobTypeError, FloatKnob, IntegerKnob, CategoricalKnob, FixedKnob, DynamicListKnob, ListKnob

class RandomAdvisor(BaseAdvisor):
    '''
    Advisor that randomly chooses knobs with no mathematical guarantee. 
    '''   
    def start(self, knob_config):
        self._knob_config = knob_config

    def propose(self):
        knobs = {
            name: self._propose(knob) 
            for (name, knob) 
            in self._knob_config.items()
        }
        return knobs
            
    def _propose(self, knob):
        u = random.uniform(0, 1)
        if isinstance(knob, FloatKnob):
            return knob.value_min + u * (knob.value_max - knob.value_min)
        elif isinstance(knob, IntegerKnob):
            return knob.value_min + int(u * (knob.value_max - knob.value_min + 1))
        elif isinstance(knob, CategoricalKnob):
            i = int(u * len(knob.values))
            return knob.values[i]
        elif isinstance(knob, FixedKnob):
            return knob.value
        elif isinstance(knob, ListKnob):
            return [self._propose(knob.items[i]) for i in range(len(knob))]
        elif isinstance(knob, DynamicListKnob):
            list_len = knob.len_min + int(u * (knob.len_max - knob.len_min + 1))
            return [self._propose(knob.items[i]) for i in range(list_len)]
        else:
            raise UnsupportedKnobTypeError(knob.__class__)

    def feedback(self, knobs, score):
        # Ignore feedback - no relevant for a random advisor
        pass
