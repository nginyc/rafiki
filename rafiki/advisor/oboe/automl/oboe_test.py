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

data = load_breast_cancer()
x = np.array(data['data'])
y = np.array(data['target'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

m = AutoLearner(p_type='classification', runtime_limit=18)
start = time.time()
m.fit(x_train, y_train)
elapsed_time = time.time() - start

# use the fitted autolearner for prediction on test set
y_predicted = m.predict(x_test)
print("prediction error: {}".format(util.error(y_test, y_predicted, 'classification')))    
print("elapsed time: {}".format(elapsed_time))
print("individual accuracies of selected models: {}".format(m.get_model_accuracy(y_test)))

print(m.get_models())