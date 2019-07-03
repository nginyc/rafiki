import xgboost as xgb
import pickle
import base64
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
        num_class = y.unique().size

        self._clf = self._build_classifier(self.n_estimators, self.min_child_weight, \
            self.max_depth, self.gamma, self.subsample, self.colsample_bytree, num_class)

        self._clf.fit(X, y)

        # Compute train accuracy
        score = self._clf.score(X, y)
        logger.log('Train accuracy: {}'.format(score))

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

        accuracy = self._clf.score(X, y)
        return accuracy

    def predict(self, queries):
        probs = [self._clf.predict_proba(pd.DataFrame.from_dict(query)).tolist()[0] for query in queries]
        return probs

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
        train_dataset_uri=os.path.join(root, 'data/titanic_train.csv'),
        test_dataset_uri=os.path.join(root, 'data/titanic_test.csv'),
        queries=[
            {'Pclass': {499: 3}, 'Age': {499: 24.0}, 'Sex_female': {499: 0}, 'Sex_male': {499: 1}}
        ],
        features=['Pclass','Sex','Age'],
        target='Survived'
    )
