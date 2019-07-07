import xgboost as xgb
import pickle
import base64
import pandas as pd
import numpy as np
from collections import OrderedDict

from pathlib import Path
import sys
import os

root = str(Path(os.path.abspath(__file__)).parents[3])
sys.path.insert(0, root)

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        IntegerKnob, FloatKnob, dataset_utils, logger
from rafiki.constants import TaskType, ModelDependency

class XgbClf(BaseModel):
    '''
    Implements a XGBoost Classifier for tabular data classification task
    '''
    @staticmethod
    def get_knob_config():
        return {
            'n_estimators': IntegerKnob(50, 200),
            'min_child_weight': IntegerKnob(1, 6),
            'max_depth': IntegerKnob(2, 8),
            'gamma': FloatKnob(0.0, 1.0, is_exp=False),
            'subsample': FloatKnob(0.5, 1.0, is_exp=False),
            'colsample_bytree': FloatKnob(0.1, 0.7, is_exp=False)
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
       
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

        num_class = y.unique().size

        self._clf = self._build_classifier(self.n_estimators, self.min_child_weight, \
            self.max_depth, self.gamma, self.subsample, self.colsample_bytree, num_class)

        self._clf.fit(X, y)

        # Compute train accuracy
        score = self._clf.score(X, y)
        logger.log('Train accuracy: {}'.format(score))

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

        accuracy = self._clf.score(X, y)
        return accuracy

    def predict(self, queries):
        decoded_queries = []
        for query in queries:
            query = [tuple(feature) for feature in query]
            decoded_queries.append(query)
        decoded_queries = [pd.DataFrame.from_dict(OrderedDict(decoded_query)) \
            for decoded_query in decoded_queries]
        probs = [self._clf.predict_proba(self._features_mapping(decoded_query)).tolist()[0] \
            for decoded_query in decoded_queries]
        return probs

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

    def _build_classifier(self, n_estimators, min_child_weight, max_depth, gamma, subsample, colsample_bytree, num_class):
        if num_class < 2:
            raise InvalidModelParamsException()
        elif num_class == 2:
            clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        ) 
        else:
            clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='multi:softmax', 
            num_class=num_class
        ) 
        return clf

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='XgbClf',
        task=TaskType.TABULAR_CLASSIFICATION,
        dependencies={
            ModelDependency.XGBOOST: '0.90'
        },
        train_dataset_path=os.path.join(root, 'data/titanic_train.zip'),
        val_dataset_path=os.path.join(root, 'data/titanic_test.zip'),
        queries=[
            [['Pclass', {'340': 1}],
            ['Sex', {'340': 'female'}],
            ['Age', {'340': 2.0}]]
        ],
    )
