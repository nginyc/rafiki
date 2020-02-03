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

from rafiki.model import BaseModel, FloatKnob, CategoricalKnob, FixedKnob,IntegerKnob,utils
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class
from sklearn.metrics import roc_auc_score, roc_curve

from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import json
import os
import tempfile
import numpy as np
import base64
import pandas as pd
import abc
import gc
import pickle
from urllib.parse import urlparse, parse_qs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

''' 
    This model is desigined for Home Credit Default Risk `https://www.kaggle
    .com/c/home-credit-default-risk` and only uses the main table 
    'application_{train|test}.csv' of this competition as dataset.
'''

class LightGBM(BaseModel):

    @staticmethod
    def get_knob_config():
        return {
            'learning_rate': FloatKnob(1e-2, 1e-1, is_exp=True),
            'num_leaves': IntegerKnob(20, 60),
            'colsample_bytree': FloatKnob(1e-1, 1),
            'subsample': FloatKnob(1e-1, 1),
            'max_depth': IntegerKnob(1, 10),
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)

    def train(self, dataset_url, features=None, target=None, exclude=None, **kwargs):
        utils.logger.define_plot('Loss Over Epochs',
                                 ['loss', 'early_stop_val_loss'],
                                 x_axis='epoch')

        self._features = features
        self._target = target

        df = pd.read_csv(dataset_url, index_col=0)
        if exclude and set(df.columns.tolist()).intersection(set(exclude)) == set(exclude):
            df = df.drop(exclude, axis=1)

        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        df = df[df['CODE_GENDER'] != 'XNA']

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(df)
        # Encode categorical features
        X = self._encoding_categorical_type(X)
        # other preprocessing
        df_train = self._preprocessing(X)

        # Cross validation model
        folds = KFold(n_splits= 10, shuffle=True)
        flag=0
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            lgb_train = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx],)
            lgb_valid = lgb.Dataset(X.iloc[valid_idx], y.iloc[valid_idx],)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'cross_entropy',
                'nthread': 4,
                'n_estimators': 10,
                'learning_rate': self.learning_rate,
                'num_leaves': self.num_leaves,
                'colsample_bytree': self.colsample_bytree,
                'subsample': self.subsample,
                'max_depth': self.max_depth,
                'verbose': -1,
            }

            abc={}
            self._model = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_train,lgb_valid],verbose_eval=100,callbacks=[lgb.record_evaluation(abc)])

            utils.logger.log(
                loss=abc['training']['cross_entropy'][-1],
                early_stop_val_loss=abc['valid_1']['cross_entropy'][-1],
                epoch=flag)
            flag+=1

    def evaluate(self, dataset_url):
        df = pd.read_csv(dataset_url, index_col=0)
        
        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        df = df[df['CODE_GENDER'] != 'XNA']

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(df)
        # Encode categorical features, no need mapping features for this model
        X = self._encoding_categorical_type(X)
        # other preprocessing
        df_train = self._preprocessing(X)

        # oof_preds = np.zeros(df.shape[0])
        oof_preds=self._model.predict(X)
        return roc_auc_score(y,oof_preds)

    def predict(self, queries):
        df = pd.DataFrame(queries)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(df)
        # Encode categorical features
        X = self._encoding_categorical_type(X)
        # other preprocessing
        df_train = self._preprocessing(X)

        predictions=self._model.predict(X)
        predicts = []
        for prediction in predictions:
            predicts.append(prediction)
        return predicts

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}
        # Save model parameters
        with tempfile.NamedTemporaryFile() as tmp:
            # Save whole model to temp h5 file
            self._model.save_model(tmp.name)
            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                h5_model_bytes = f.read()

        data_config_bytes = pickle.dumps([self._features, self._target])

        params['h5_model_base64'] = base64.b64encode(h5_model_bytes).decode('utf-8')
        params['data_config_base64'] = base64.b64encode(data_config_bytes).decode('utf-8')

        return params

    def load_parameters(self, params):
        # Load model parameters
        h5_model_base64 = params.get('h5_model_base64', None)
        data_config_base64 = params.get('data_config_base64', None)

        data_config_bytes = base64.b64decode(data_config_base64.encode('utf-8'))
        self._features, self._target = pickle.loads(data_config_bytes)

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)
            # Load model from temp file
            self._model =lgb.Booster(model_file=tmp.name)

    def _extract_xy(self, data):
        if self._target is None:
            self._target = 'TARGET'
        y = data[self._target] if self._target in data else None

        if self._features is None:
            X = data.drop(self._target, axis=1)
            self._features = list(X.columns)
        else:
            X = data[self._features]

        return (X, y)

    def _encoding_categorical_type(self, df):
        # Categorical features with Binary encode (0 or 1; two categories)
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        return df

    def _preprocessing(self,df):
        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        # Some simple new features (percentages)
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        return df

    def _features_mapping(self, df):
        pass

    def _build_model(self):
        pass


if __name__ == '__main__':
    curpath = os.path.join(os.environ['HOME'], 'rafiki')
    os.environ.setdefault('WORKDIR_PATH', curpath)
    os.environ.setdefault('PARAMS_DIR_PATH', os.path.join(curpath, 'params'))

    train_set_url = os.path.join(curpath, 'data', 'application_train_index.csv')
    valid_set_url = train_set_url
    test_set_url = os.path.join(curpath, 'data', 'application_test_index.csv')

    test_queries = pd.read_csv(test_set_url, index_col=0).iloc[:5]
    test_queries = json.loads(test_queries.to_json(orient='records'))


    test_model_class(
        model_file_path=__file__,
        model_class='LightGBM',
        task=None,
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0',
            'lightgbm': '2.3.0',
        },
        train_dataset_path=train_set_url,
        val_dataset_path=valid_set_url,
        train_args={
            'target': 'TARGET',
            'exclude': ['SK_ID_CURR'],
        },
        queries=test_queries,
    )
