import numpy as np
import pandas as pd
import json 
import pickle
import base64
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rafiki.model import BaseModel, IntegerKnob, CategoricalKnob, FloatKnob, logger
from rafiki.model.dev import test_model_class
from rafiki.constants import ModelDependency

class SVCClf(BaseModel):
    '''
    Implements a C-Support Vector Classifier for classification task using Pima Indian Diabetes dataset.
    '''
    @staticmethod
    def get_knob_config():
        return {
            'C': IntegerKnob(2,3),
            'kernel': CategoricalKnob(['poly','rbf','linear']),
            'degree': IntegerKnob(2,3),
            'gamma': CategoricalKnob(['scale','auto']),
            'coef0': FloatKnob(0.0, 0.1),
            'shrinking': CategoricalKnob([True,False]),
            'tol': FloatKnob(1e-03, 1e-01, is_exp=True),
            'decision_function_shape' : CategoricalKnob(['ovo','ovr']),
            'probability' : CategoricalKnob([True,False]),
            
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.C, self.kernel, self.degree, self.gamma, self.coef0, self.shrinking, self.tol, self.decision_function_shape, self.probability)

       
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
        X = StandardScaler().fit_transform(data)
        return X
    

    def _build_classifier(self, C, kernel, degree, gamma, coef0, shrinking, tol, decision_function_shape, probability):
        clf = SVC(
            C=C,
            kernel=kernel,
            degree = degree,
            gamma = gamma,
            coef0 = coef0,
            shrinking = shrinking,
            tol = tol,
            decision_function_shape = decision_function_shape,
            probability = probability,
        )
        return clf

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='SVCClf',
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
