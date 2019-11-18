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

import nltk, re, pprint
from nltk.tokenize import word_tokenize
import pickle
import base64
import numpy as np
import argparse
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter

from rafiki.model import BaseModel, utils, FixedKnob, FloatKnob
from rafiki.model.dev import test_model_class


class CRFClf(BaseModel):
    '''
    Implements Conditional Random Field classifier for POS tagging, using treebank dataset & universal tagset. 
    code credit to 'https://github.com/AiswaryaSrinivas/DataScienceWithPython/blob/master/CRF%20POS%20Tagging.ipynb'
    '''
    @staticmethod
    def get_knob_config():
        return {
            'c1': FloatKnob(0.001, 0.01),
            'c2': FloatKnob(0.01, 0.1),
            'max_iterations': FixedKnob(10)
        }

    
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.c1, self.c2, self.max_iterations)
        
        
    def train(self, dataset_path, **kwargs):
        with open(dataset_path, "rb") as fp:
            data=pickle.load(fp)
        
        X_train,y_train=self.prepareData(data)

        self._clf.fit(X_train, y_train)
        
        # Compute train accuracy
        y_pred = self._clf.predict(X_train)        
        accuracy = metrics.flat_f1_score(y_train,y_pred,average='weighted',labels=self._clf.classes_)
        utils.logger.log('Train accuracy: {}'.format(accuracy))
        
    
    def evaluate(self, dataset_path):
        with open(dataset_path, "rb") as fp:
            data=pickle.load(fp)
        
        X_test,y_test=self.prepareData(data)
        
        # Compute test accuracy
        y_pred=self._clf.predict(X_test)
        acc = metrics.flat_f1_score(y_test,y_pred,average='weighted',labels=self._clf.classes_)
        return acc

    
    def predict(self, queries):
        sents_tokens = queries
        (sents_tags) = self._clf.predict(sents_tokens)
        return sents_tags

    
    def dump_parameters(self):
        params = {}
        # Save model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64

        return params

    
    def load_parameters(self, params):
        # Load model parameters
        clf_base64 = params['clf_base64']
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)

        
    def features(self, sentence, index):
    ### sentence is of the form [w1,w2,w3,..], index is the position of the word in the sentence
        return {
            'is_first_capital':int(sentence[index][0].isupper()),
            'is_first_word': int(index==0),
            'is_last_word':int(index==len(sentence)-1),
            'is_complete_capital': int(sentence[index].upper()==sentence[index]),
            'prev_word':'' if index==0 else sentence[index-1],
            'next_word':'' if index==len(sentence)-1 else sentence[index+1],
            'is_numeric':int(sentence[index].isdigit()),
            'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])',sentence[index])))),
            'prefix_1':sentence[index][0],
            'prefix_2': sentence[index][:2],
            'prefix_3':sentence[index][:3],
            'prefix_4':sentence[index][:4],
            'suffix_1':sentence[index][-1],
            'suffix_2':sentence[index][-2:],
            'suffix_3':sentence[index][-3:],
            'suffix_4':sentence[index][-4:],
            'word_has_hyphen': 1 if '-' in sentence[index] else 0
        }

    
    def untag(self, sentence):
        return [word for word,tag in sentence]

    
    def prepareData(self, tagged_sentences):
        X,y=[],[]
        for sentences in tagged_sentences:
            X.append([self.features(self.untag(sentences), index) for index in range(len(sentences))])
            y.append([tag for word,tag in sentences])
        return X,y
    
    
    def _build_classifier(self, c1, c2, max_iterations):
        c1 =self._knobs.get('c1')
        c2 =self._knobs.get('c2')
        max_iterations =self._knobs.get('max_iterations')
        
        clf = CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )
     
        return clf

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='CRFClf',
        task='POS_TAGGING',
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0',
            ModelDependency.NLTK: '3.4.5',
            ModelDependency.SKLEARN_CRFSUITE: '0.3.6'
        },
        train_dataset_path='data/ptb_train.txt',
        val_dataset_path='data/ptb_test.txt',
        queries=[
            ['Ms.', 'Haag', 'plays', 'Elianti', '18', '.'],
            ['The', 'luxury', 'auto', 'maker', 'last', 'year', 'sold', '1,214', 'cars', 'in', 'the', 'U.S.']
        ]
    )
    
    