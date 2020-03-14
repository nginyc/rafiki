import numpy as np
import pandas as pd
import pickle
import base64  
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from rafiki.model import BaseModel, IntegerKnob, CategoricalKnob, FloatKnob, logger
from rafiki.model.dev import test_model_class
from rafiki.constants import ModelDependency

class LogisticRegClf(BaseModel):
    '''
    Implements a Logistic Regression Classifier for classification task using Pima Indian Diabetes dataset.
    '''
    @staticmethod
    def get_knob_config():
        return {
            'penalty': CategoricalKnob(['l1', 'l2']),
            'tol': FloatKnob(0.0001, 0.001),
            'C': IntegerKnob(4,15),
            'fit_intercept': CategoricalKnob([True,False]),
            'solver': CategoricalKnob(['lbfgs', 'liblinear']),
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.penalty, self.tol, self.C, self.fit_intercept, self.solver)

       
    def train(self, dataset_path, **kwargs):       
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        df = pd.read_csv(csv_path)
        data = self.median_dataset(df)
        
        X_train = self.prepare_X(data)
        y_train = data.iloc[:, -1]

        self._clf.fit(X_train, y_train)

        # Compute train accuracy
        score = self._clf.score(X_train, y_train)
        logger.log('Train accuracy: {}'.format(score))
        

    def evaluate(self, dataset_path):
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        df = pd.read_csv(csv_path)
        data = self.median_dataset(df)

        X_val = self.prepare_X(data)
        y_val = data.iloc[:, -1]
        
        # Compute test accuracy
        accuracy = self._clf.score(X_val, y_val)
        return accuracy

    
    def predict(self, queries):
        queries = [pd.DataFrame(query, index=[0]) for query in queries]
        data = self.median_dataset(queries)
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

        return params
    

    def load_parameters(self, params):
        # Load model parameters
        assert 'clf_base64' in params
        clf_base64 = params['clf_base64']
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)
        
        
    def median_dataset(self, df):
        #replace zero values by median so that 0 will not affect median.
        df.Glucose.replace(0, np.nan, inplace=True)
        df.Glucose.replace(np.nan, df['Glucose'].median(), inplace=True)
        df.BloodPressure.replace(0, np.nan, inplace=True)
        df.BloodPressure.replace(np.nan, df['BloodPressure'].median(), inplace=True)
        df.SkinThickness.replace(0, np.nan, inplace=True)
        df.SkinThickness.replace(np.nan, df['SkinThickness'].median(), inplace=True)
        df.Insulin.replace(0, np.nan, inplace=True)
        df.Insulin.replace(np.nan, df['Insulin'].median(), inplace=True)
        df.BMI.replace(0, np.nan, inplace=True)
        df.BMI.replace(np.nan, df['BMI'].median(), inplace=True)
        return df

    
    def prepare_X(self, data):
        X = data.iloc[:, :-1]
        #Reduce data dimensionality
        pca = PCA()
        return pca.fit_transform(X)


    def _build_classifier(self, penalty, tol, C, fit_intercept, solver):
        clf = LogisticRegression(
            penalty = penalty,
            tol = tol,
            C=C,
            fit_intercept = fit_intercept,
            solver = solver,
        )
        return clf

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='LogisticRegClf',
        task='TABULAR_CLASSIFICATION',
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0'
        },
        train_dataset_path='data/diabetes.csv',
        val_dataset_path='data/diabetes.csv',
        queries=[
            { 'Pregnancies': 3, 'Glucose': '130', 'BloodPressure': 92, 'SkinThickness': 30, 'Insulin': 90, 'BMI': 30.4, 'Age': 40 }
        ]
    )
