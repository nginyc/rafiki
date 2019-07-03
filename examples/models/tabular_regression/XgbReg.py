import xgboost as xgb
from sklearn.metrics import mean_squared_error
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
       
    def train(self, dataset_path, features=None, target=None):
        dataset = dataset_utils.load_dataset_of_tabular(dataset_path)
        data = dataset.data
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
        self._clf.fit(X, y)

        # Compute train root mean square error
        preds = self._clf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        logger.log('Train RMSE: {}'.format(rmse))

    def evaluate(self, dataset_path, features=None, target=None):
        dataset = dataset_utils.load_dataset_of_tabular(dataset_path)
        data = dataset.data
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
            {'CRIM': {370: 6.53876}, 'ZN': {370: 0.0}, 'INDUS': {370: 18.1}, 'CHAS': {370: 1.0}, 
            'NOX': {370: 0.631}, 'RM': {370: 7.016}, 'AGE': {370: 97.5}, 'DIS': {370: 1.2024}, 
            'RAD': {370: 24.0}, 'TAX': {370: 666.0}, 'PTRATIO': {370: 20.2}, 'B': {370: 392.05}}
        ],
        train_dataset_uri=os.path.join(root, 'data/boston_train.csv'),
        test_dataset_uri=os.path.join(root, 'data/boston_test.csv')
    )
