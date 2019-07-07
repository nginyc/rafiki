import xgboost as xgb
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
import pickle
import base64
import numpy as np
import pandas as pd

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
        X = self._encoding_categorical_type(X)

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
        X = self._encoding_categorical_type(X)

        preds = self._clf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return rmse

    def predict(self, queries):
        decoded_queries = []
        for query in queries:
            query = [tuple(feature) for feature in query]
            decoded_queries.append(query)
        decoded_queries = [pd.DataFrame.from_dict(OrderedDict(decoded_query)) \
            for decoded_query in decoded_queries]
        results = [self._clf.predict(self._features_mapping(decoded_query)).tolist()[0] \
            for decoded_query in decoded_queries]
        return results

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}

        # Put model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64
        params['encoding_dict'] = self._encoding_dict

        return params

    def load_parameters(self, params):
        # Load model parameters
        clf_base64 = params['clf_base64']
        if clf_base64 is None:
            raise InvalidModelParamsException()
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)
        self._encoding_dict = params['encoding_dict']

    def _encoding_categorical_type(self, cols):
        # Apply label encoding for those categorical columns
        cat_cols = list(filter(lambda x: cols[x].dtype == 'object', cols.columns))
        encoded_cols = pd.DataFrame({col: cols[col].astype('category').cat.codes \
            if cols[col].dtype == 'object' else cols[col] for col in cols}, index=cols.index)

        # Recover the missing elements (Use XGBoost to automatically handle them)
        encoded_cols = encoded_cols.replace(to_replace = -1, value = np.nan)

        # Generate the dict that maps categorical features to numerical
        encoding_dict = {col: {cat: n for n, cat in enumerate(cols[col].astype('category'). \
            cat.categories)} for col in cat_cols}
        self._encoding_dict = encoding_dict

        return encoded_cols

    def _features_mapping(self, df):
        # Encode the categorical features with pre saved encoding dict
        cat_cols = list(filter(lambda x: df[x].dtype == 'object', df.columns))
        df_temp = df.copy()
        for col in cat_cols:
            df_temp[col] = df[col].map(self._encoding_dict[col])
        df = df_temp
        return df

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
             [['density', {241: 1.0207}],
             ['age', {241: 65}],
             ['weight', {241: 224.5}],
             ['height', {241: 68.25}],
             ['neck', {241: 38.8}],
             ['chest', {241: 119.6}],
             ['abdomen', {241: 118.0}],
             ['hip', {241: 114.3}],
             ['thigh', {241: 61.3}],
             ['knee', {241: 42.1}],
             ['ankle', {241: 23.4}],
             ['biceps', {241: 34.9}],
             ['forearm', {241: 30.1}],
             ['wrist', {241: 19.4}]]
        ],
        train_dataset_path=os.path.join(root, 'data/bodyfat_train.zip'),
        val_dataset_path=os.path.join(root, 'data/bodyfat_test.zip')
    )
