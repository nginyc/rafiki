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

import pickle
import base64
import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import PassiveAggressiveClassifier

from rafiki.model import BaseModel, IntegerKnob, FloatKnob, CategoricalKnob, logger
from rafiki.model.dev import test_model_class
from rafiki.constants import ModelDependency

class PassiveAClf(BaseModel):
    '''
    Implements a Passive Aggressive Classifier for classification task using Pima Indian Diabetes dataset.
    '''
    @staticmethod
    def get_knob_config():
        return {
            'C': FloatKnob(1.0,1.5),
            'tol' : FloatKnob(1e-03, 1e-01, is_exp=True),
            'validation_fraction': FloatKnob(0.01, 0.1),
            'n_iter_no_change': IntegerKnob(3, 5),
            'shuffle': CategoricalKnob([True,False]),
            'loss': CategoricalKnob(['hinge', 'squared_hinge']),
            'random_state': IntegerKnob(1, 2),
            'warm_start': CategoricalKnob([True,False]),
            'average': IntegerKnob(1, 5),
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.C, self.tol, self.validation_fraction, self.n_iter_no_change, self.shuffle, self.loss, self.random_state, self.warm_start, self.average)

    
    def train(self, dataset_path, features=None, target=None, **kwargs):
        # Record features & target
        self._features = features
        self._target = target
        
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        X = self.prepare_X(X)
        
        self._clf.fit(X, y)

        # Compute train accuracy
        score = self._clf.score(X, y)
        logger.log('Train accuracy: {}'.format(score))

        
    def evaluate(self, dataset_path):
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        X = self.prepare_X(X)

        accuracy = self._clf.score(X, y)
        return accuracy

    
    def predict(self, queries):
        queries = [pd.DataFrame(query, index=[0]) for query in queries]
        data = self.prepare_X(queries)
        probs = self._clf.predict_proba(data)
        return probs.tolist()
    
    
    def destroy(self):
        pass

    
    def dump_parameters(self):
        params = {}

        # Put model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64
        params['features'] = json.dumps(self._features)
        params['target'] = self._target

        return params

    def load_parameters(self, params):
        # Load model parameters
        assert 'clf_base64' in params
        clf_base64 = params['clf_base64']
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))

        self._clf = pickle.loads(clf_bytes)
        self._features = json.loads(params['features'])
        self._target = params['target']

        
    def _extract_xy(self, data):
        features = self._features
        target = self._target

        if features is None:
            X = data.iloc[:,:-1]
        else:
            X = data[features]
            
        if target is None:
            y = data.iloc[:,-1]
        else:
            y = data[target]

        return (X, y)
        
        
    def median_dataset(self, df):
        #replace zero values by median so that 0 will not affect median.
        for col in df.columns:
            df[col].replace(0, np.nan, inplace=True)
            df[col].fillna(df[col].median(), inplace=True)
        return df

    
    def prepare_X(self, df):
        data = self.median_dataset(df)
        X = PolynomialFeatures(interaction_only=True).fit_transform(data).astype(int)
        return X
    

    def _build_classifier(self, C, tol, validation_fraction, n_iter_no_change, shuffle, loss, random_state, warm_start, average):
        clf = PassiveAggressiveClassifier(
            C=C,
            tol = tol,
            validation_fraction = validation_fraction,
            n_iter_no_change = n_iter_no_change,
            shuffle = shuffle,
            loss = loss,
            random_state = random_state,
            warm_start = warm_start,
            average = average,
        )
        return clf

    
if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='PassiveAClf',
        task='TABULAR_CLASSIFICATION',
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0'
        },
        train_dataset_path='data/diabetes_train.csv',
        val_dataset_path='data/diabetes_val.csv',
        train_args={
            'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'],
            'target':'Outcome'
        },
        queries=[
            { 'Pregnancies': 3, 'Glucose': '130', 'BloodPressure': 92, 'SkinThickness': 30, 'Insulin': 90, 'BMI': 30.4, 'Age': 40 }
        ]
    )
