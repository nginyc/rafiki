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

import math
import json

from rafiki.model import BaseModel, utils, FixedKnob
from rafiki.model.dev import test_model_class

class BigramHmm(BaseModel):
    '''
    Implements Bigram Hidden Markov Model (HMM) for POS tagging
    '''
    @staticmethod
    def get_knob_config():
        return {
            'min_value': FixedKnob(-9999999999) # Min numeric value
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._min_value = knobs['min_value']

    def train(self, dataset_path, **kwargs):
        dataset = utils.dataset.load_dataset_of_corpus(dataset_path)
        (sents_tokens, sents_tags) = zip(*[zip(*sent) for sent in dataset])
        self._num_tags = dataset.tag_num_classes[0]
        (self._trans_probs, self._emiss_probs) = self._compute_probs(self._num_tags, sents_tokens, sents_tags)
        utils.logger.log('No. of tags: {}'.format(self._num_tags))

    def evaluate(self, dataset_path):
        dataset = utils.dataset.load_dataset_of_corpus(dataset_path)
        (sents_tokens, sents_tags) = zip(*[zip(*sent) for sent in dataset])
        (sents_pred_tags) = self._tag_sents(self._num_tags, sents_tokens, self._trans_probs, self._emiss_probs)
        acc = self._compute_accuracy(sents_tags, sents_pred_tags)
        return acc

    def predict(self, queries):
        sents_tokens = queries
        (sents_tags) = self._tag_sents(self._num_tags, sents_tokens, self._trans_probs, self._emiss_probs)
        return sents_tags

    def dump_parameters(self):
        params = {}
        params['emiss_probs'] = json.dumps(self._emiss_probs)
        params['trans_probs'] = json.dumps(self._trans_probs)
        params['num_tags'] = self._num_tags
        return params

    def load_parameters(self, params):
        self._emiss_probs = json.loads(params['emiss_probs'])
        self._trans_probs = json.loads(params['trans_probs'])
        self._num_tags = params['num_tags']

    def _compute_accuracy(self, sents_tags, sents_pred_tags):
        total = 0
        correct = 0

        for (tags, pred_tags) in zip(sents_tags, sents_pred_tags):
            for (tag, pred_tag) in zip(tags, pred_tags):
                total += 1
                if tag == pred_tag: correct += 1

        return correct / total

    def _compute_probs(self, num_tags, sents_tokens, sents_tags):

        # Total number of states in HMM as tags
        T = num_tags + 2 # Last 2 for START & END tags
        START = num_tags # <s>
        END = num_tags + 1 # </s>

        # Unigram (tag i) counts
        uni_counts = [0 for i in range(T)]

        # Bigram (tag i, tag j) counts
        bi_counts = [[0 for j in range(T)] for i in range(T)]
        
        # Counts for (tag i, word w) as [{ w -> count }]
        word_counts = [{} for i in range(T)]

        # For each sentence
        for (tokens, tags) in zip(sents_tokens, sents_tags):
            uni_counts[START] += 1

            # Run through sentence and update counts
            prev_tag = None
            for (word, tag) in zip(tokens, tags):
                if prev_tag is None:
                    bi_counts[START][tag] += 1
                else:
                    bi_counts[prev_tag][tag] += 1

                uni_counts[tag] += 1
                word_counts[tag][word] = word_counts[tag].get(word, 0) + 1
                prev_tag = tag

            uni_counts[END] += 1
            
            # Account for last bigram with </s>
            if len(tokens) > 0:
                last_tag = tags[-1]
                bi_counts[last_tag][END] += 1
          
        # Transition function (tag i, tag j) -> <log prob of transition from state i to j>
        trans_probs = [[0 for j in range(T)] for i in range(T)]
        for i in range(T):
            for j in range(T):
                if bi_counts[i][j] == 0:
                    trans_probs[i][j] = self._min_value
                else:
                    trans_probs[i][j] = math.log(bi_counts[i][j] / uni_counts[i])

        # Emission function as (tag i, word w) -> <log prob of emitting word w at state i>
        emiss_probs = [{} for i in range(T)]
        for i in range(T):
            for w in word_counts[i]:
                emiss_probs[i][w] = math.log(word_counts[i][w] / uni_counts[i])
        
        return (trans_probs, emiss_probs)

    def _tag_sents(self, num_tags, sents_tokens, trans_probs, emiss_probs):
        sents_tags = []

        T = num_tags + 2 # Last 2 for START & END tags
        START = num_tags # <s>
        END = num_tags + 1 # </s>

        for tokens in sents_tokens:
            if len(tokens) == 0:
                continue

            # Maximum log probabilities for sentence up to word w, where the last word's tag is i
            log_probs = [[None for i in range(T)] for w in range(len(tokens))]

            # Backpointers to previous best tags for log probabilities
            backpointers = [[None for i in log_probs[0]] for w in log_probs]

            # Process 1st word that is conditioned on <s>
            for i in range(T):
                trans = trans_probs[START][i]
                emiss = emiss_probs[i].get(tokens[0], self._min_value)
                log_probs[0][i] = trans + emiss

            # For each word w after the 1st word
            for w in range(1, len(tokens)):
                # For each tag i
                for i in range(T):
                    # For each prev tag j
                    for j in range(T):
                        # Compute probability for (tag j, tag i) for sentence up to word w
                        trans = trans_probs[j][i]
                        emiss = emiss_probs[i].get(tokens[w], self._min_value)
                        prob = log_probs[w - 1][j] + trans + emiss
                        if log_probs[w][i] is None or prob > log_probs[w][i]:
                            log_probs[w][i] = prob
                            backpointers[w][i] = j

            # Compare probabilities with </s> across all tags of last word
            backpointer = None
            best_prob = None
            for i in range(T):
                trans = trans_probs[i][END]
                prob = log_probs[-1][i] + trans
                if best_prob is None or prob > best_prob:
                    best_prob = prob
                    backpointer = i

            # Traverse backpointers to get most probable tags
            cur = backpointer
            w = len(tokens) - 1
            sent_tags = []
            while cur is not None:
                sent_tags.append(cur)
                cur = backpointers[w][cur]
                w -= 1
            
            sent_tags.reverse()
            sents_tags.append(sent_tags)

        return sents_tags

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='BigramHmm',
        task='POS_TAGGING',
        dependencies={},
        train_dataset_path='data/ptb_train.zip',
        val_dataset_path='data/ptb_val.zip',
        queries=[
            ['Ms.', 'Haag', 'plays', 'Elianti', '18', '.'],
            ['The', 'luxury', 'auto', 'maker', 'last', 'year', 'sold', '1,214', 'cars', 'in', 'the', 'U.S.']
        ]
    )
    
    