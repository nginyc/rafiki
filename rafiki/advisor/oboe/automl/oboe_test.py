# necessary modules
import sys
import pandas as pd
import os
import time
import numpy as np
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

#Oboe modules; this will be simplified when Oboe becomes pip installable
from auto_learner import AutoLearner
import util

#import scikit-learn modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('/Users/pro/Desktop/rafiki_fork/data/titanic_train.csv')
features = ['Pclass', 'Sex', 'Age']
target = 'Survived'
if features is None:
    x = data.iloc[:,:-1]
else:
    x = data[features]
    
if target is None:
    y = data.iloc[:,-1]
else:
    y = data[target]

x = np.array(util.encoding_categorical_type(x))
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

autolearner_kwargs = {
    'p_type': 'classification',
    'runtime_limit': 30,
    'algorithms': ['SkRf', 'XgbClf']
}

m = AutoLearner(**autolearner_kwargs)
m.fit(x_train, y_train)

y_predicted = m.predict(x_test)
best_algos = m.get_best_models(y_test)
print(best_algos)