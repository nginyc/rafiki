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

###################################
# preprocessing for structured data
###################################

def preprocess_data(df):
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # del test_df
    gc.collect()
    return df


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
        super().__init__(**knobs)
        self._knobs = knobs

    def train(self, dataset_url, **kwargs):

        self.data_addr = dataset_url
        utils.logger.define_plot('Loss Over Epochs',
                                 ['loss', 'early_stop_val_loss'],
                                 x_axis='epoch')
        target='TARGET'
        df = pd.read_csv(dataset_url, sep=',', index_col=0)
        train_df = preprocess_data(df)

        self.raw_columns = list(df.columns)
        del df

        # Cross validation model
        folds = KFold(n_splits= 10, shuffle=True)
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        feats = [f for f in train_df.columns if f not in [target]]
        flag=0
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[target])):

            train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]
            lgb_train = lgb.Dataset(train_x, train_y,)
            lgb_valid = lgb.Dataset(valid_x, valid_y,)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'cross_entropy',
                'nthread': 4,
                'n_estimators': 10,
                'learning_rate': 0.02,
                'num_leaves': self._knobs['num_leaves'],
                'colsample_bytree': self._knobs['colsample_bytree'],
                'subsample': self._knobs['subsample'],
                'max_depth': self._knobs['max_depth'],
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
        target='TARGET'
        df = pd.read_csv(self.data_addr,sep=',', index_col=0).iloc[:1000]
        df = preprocess_data(df)

        oof_preds = np.zeros(df.shape[0])
        feats=[f for f in df.columns if f not in [target]]
        valid_x=df[feats]
        valid_y=df[target]
        oof_preds=self._model.predict(valid_x)
        return roc_auc_score(valid_y,oof_preds)


    def predict(self, queries):
        df = pd.DataFrame(queries)
        self.raw_columns.remove('TARGET')
        df.columns = self.raw_columns
        test_df = preprocess_data(df)

        predictions=self._model.predict(test_df)
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

            data_config = pickle.dumps(self.raw_columns)
            params['h5_model_base64'] = base64.b64encode(h5_model_bytes).decode('utf-8')
            params['data_config'] = base64.b64encode(data_config).decode('utf-8')

        return params

    def load_parameters(self, params):
        # Load model parameters
        h5_model_base64 = params.get('h5_model_base64', None)
        data_config_byte = params.get('data_config', None)
        self.raw_columns = pickle.loads(base64.b64decode(data_config_byte.encode('utf-8')))

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)

            # Load model from temp file
            self._model =lgb.Booster(model_file=tmp.name)

    def _build_model(self):
        pass


if __name__ == '__main__':
    curpath = os.path.join(os.environ['HOME'], 'Documents', 'Github', 'rafiki')
    os.environ.setdefault('WORKDIR_PATH', curpath)
    os.environ.setdefault('PARAMS_DIR_PATH', os.path.join(curpath, 'params'))

    train_set_url = os.path.join(curpath, 'data', 'application_train_index.csv')
    valid_set_url = train_set_url
    test_set_url = os.path.join(curpath, 'data', 'application_test_index.csv')
    df_test = pd.read_csv(test_set_url, index_col=0)

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
        # budget={'MODEL_TRAIL_COUNT': 1},
        queries=df_test.values.tolist(),
    )
