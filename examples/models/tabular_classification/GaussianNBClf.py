import numpy as np
import pandas as pd
import pickle
import base64
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from rafiki.model import BaseModel, IntegerKnob, FloatKnob, logger
from rafiki.model.dev import test_model_class
from rafiki.constants import ModelDependency

class GaussianClf(BaseModel):
    '''
    Implements Gaussian Naive Bayes Classifier for tabular data classification task
    '''
    @staticmethod
    def get_knob_config():
        return {
            'var_smoothing': FloatKnob(1e-07, 1e-05, is_exp=True),          
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.var_smoothing)

       
    def train(self, dataset_path, **kwargs):       
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)
        
        X_train = self.prepare_X(data)
        y_train = data.iloc[:, -1]

        self._clf.fit(X_train, y_train)

        # Compute train accuracy
        score = self._clf.score(X_train, y_train)
        logger.log('Train accuracy: {}'.format(score))
        

    def evaluate(self, dataset_path):
        csv_path = dataset_path
        data = pd.read_csv(csv_path)
        
        X_val = self.prepare_X(data)
        y_val = data.iloc[:, -1]

        accuracy = self._clf.score(X_val, y_val)
        return accuracy

    
    def predict(self, queries):
        queries = [pd.DataFrame(query, index=[0]) for query in queries]    
        probs = self._clf.predict_proba(queries)
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
        
        
    def prepare_X(self, data):
        sc = StandardScaler()
        return sc.fit_transform(data)
    
    
    def _build_classifier(self, var_smoothing):
        clf = GaussianNB(priors=None, var_smoothing=var_smoothing)
        return clf

    
if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='GaussianClf',
        task='TABULAR_CLASSIFICATION',
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0'
        },
        train_dataset_path='data/heart_train.csv',
        val_dataset_path='data/heart_test.csv',
        queries=[
            { 'age': 50, 'Sex': '0', 'cp': 3, 'trestbps': 130, 'chol': 220, 'fbs': 1, 'restecg': 0, 'thalach': 170, 'exang': 1, 'oldpeak': 1.7, 'slope': 2, 'ca': 0, 'thal': 3 }
        ]
    )