import xgboost as xgb
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
import pickle
import base64
import numpy as np
import pandas as pd
import category_encoders as ce

from pathlib import Path
import sys
import os

root = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.insert(0, root)

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        IntegerKnob, FloatKnob, dataset_utils, logger
from rafiki.constants import TaskType, ModelDependency

class XgbReg(BaseModel):
    '''
    Implements a XGBoost Regressor for tabular data regression task
    '''
    @staticmethod
    def get_knob_config():
        return {
            'n_estimators': IntegerKnob(50, 200),
            'min_child_weight': IntegerKnob(1, 6),
            'max_depth': IntegerKnob(1, 10),
            'gamma': FloatKnob(0.0, 1.0, is_exp=False),
            'subsample': FloatKnob(0.5, 1.0, is_exp=False),
            'colsample_bytree': FloatKnob(0.1, 0.7, is_exp=False)
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.n_estimators, self.min_child_weight, \
            self.max_depth, self.gamma, self.subsample, self.colsample_bytree)
       
    def train(self, dataset_path):
        dataset = dataset_utils.load_dataset_of_tabular(dataset_path)
        data = dataset.data
        table_meta = dataset.table_meta

        if table_meta != {}:
            features = table_meta['features']
            target = table_meta['target']
        else:
            features = None
            target = None

        if features is None:
            X = data.iloc[:,:-1]
        else:
            X = data[features]
        if target is None:
            y = data.iloc[:,-1]
        else:
            y = data[target]

        # Encode categorical features
        X = self._category_encoding_type(X, y)
        self._clf.fit(X, y)

        # Compute train root mean square error
        preds = self._clf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        logger.log('Train RMSE: {}'.format(rmse))

    def evaluate(self, dataset_path):
        dataset = dataset_utils.load_dataset_of_tabular(dataset_path)
        data = dataset.data
        table_meta = dataset.table_meta

        if table_meta != {}:
            features = table_meta['features']
            target = table_meta['target']
        else:
            features = None
            target = None

        if features is None:
            X = data.iloc[:,:-1]
        else:
            X = data[features]
        if target  is None:
            y = data.iloc[:,-1]
        else:
            y = data[target]

        # Encode categorical features
        X = self._category_encoding_type(X, y)
        preds = self._clf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return rmse

    def predict(self, queries):
        results = [self._clf.predict(pd.DataFrame.from_dict(query)).tolist()[0] for query in queries]
        return results

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}

        # Put model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64

        return params

    def load_parameters(self, params):
        # Load model parameters
        clf_base64 = params['clf_base64']
        if clf_base64 is None:
            raise InvalidModelParamsException()
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)

    def _category_encoding_type(self, cols, target):
        # Apply target encoding for those categorical columns that have too many features
        cols_target = list(filter(lambda x: cols[x].dtype == 'object' and cols[x].unique().size > 5, cols.columns))
        if cols_target != []:
            ce_target = ce.TargetEncoder(cols = cols_target)
            ce_target.fit(cols, target)
            cols = ce_target.transform(cols, target)
        # Apply one-hot encoding for the rest categorical columns
        cols = pd.get_dummies(cols)
        return cols

    def _build_classifier(self, n_estimators, min_child_weight, max_depth, gamma, subsample, colsample_bytree):
        clf = xgb.XGBRegressor(
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        ) 
        return clf

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='XgbReg',
        task=TaskType.TABULAR_REGRESSION,
        dependencies={
            ModelDependency.XGBOOST: '0.90'
        },
        queries=[
            OrderedDict([('density', {241: 1.0207}),
             ('age', {241: 65}),
             ('weight', {241: 224.5}),
             ('height', {241: 68.25}),
             ('neck', {241: 38.8}),
             ('chest', {241: 119.6}),
             ('abdomen', {241: 118.0}),
             ('hip', {241: 114.3}),
             ('thigh', {241: 61.3}),
             ('knee', {241: 42.1}),
             ('ankle', {241: 23.4}),
             ('biceps', {241: 34.9}),
             ('forearm', {241: 30.1}),
             ('wrist', {241: 19.4})])
        ],
        train_dataset_uri=os.path.join(root, 'data/bodyfat_train.zip'),
        test_dataset_uri=os.path.join(root, 'data/bodyfat_test.zip')
    )
