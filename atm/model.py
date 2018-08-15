"""
.. module:: wrapper
   :synopsis: Model around classification method.

"""
from __future__ import absolute_import, division, unicode_literals

import logging
import re
import time
from builtins import object
from collections import defaultdict
from importlib import import_module

import numpy as np
import pandas as pd
from past.utils import old_div
from sklearn import decomposition
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                              ExpSineSquared, Matern,
                                              RationalQuadratic)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .constants import *
from .encoder import DataEncoder, MetaData
from .method import Method
from .metrics import cross_validate_pipeline, test_pipeline

import methods

# load the library-wide logger
logger = logging.getLogger('atm')


class Model(object):
    """
    This class contains everything needed to run an end-to-end ATM classifier
    pipeline. It is initialized with a set of parameters and trained like a
    normal sklearn model. This class can be pickled and saved to disk, then
    unpickled outside of ATM and used to classify new datasets.
    """
    # these are special keys that are used for general purpose
    # things like scaling, normalization, PCA, etc
    SCALE = "_scale"
    WHITEN = "_whiten"
    MINMAX = "_scale_minmax"
    PCA = "_pca"
    PCA_DIMS = "_pca_dimensions"

    # list of all such keys
    ATM_KEYS = [SCALE, WHITEN, MINMAX, PCA, PCA_DIMS]

    # number of folds for cross-validation (arbitrary, for speed)
    N_FOLDS = 5

    def __init__(self, method, params, judgment_metric, 
                 testing_ratio=0.3, verbose_metrics=False):
        """
        Parameters:
            method: the short method code (as defined in constants.py) or path
                to method json
            judgment_metric: string that indicates which metric should be
                optimized for.
            params: parameters passed to the sklearn classifier constructor
        """
        # configuration & database
        self.method = method
        self.params = params
        self.judgment_metric = judgment_metric
        self.testing_ratio = testing_ratio
        self.verbose_metrics = verbose_metrics

        # load the classifier method's class
        path = Method(method).class_path.split('.')
        mod_str, cls_str = '.'.join(path[:-1]), path[-1]
        mod = import_module(mod_str)
        self.class_ = getattr(mod, cls_str)

        # pipelining
        self.pipeline = None

        # persistent random state
        self.random_state = np.random.randint(1e7)

    def make_pipeline(self):
        """
        Makes the classifier as well as scaling or dimension reduction steps.
        """
        # create a list of steps, starting with the data encoder
        steps = []

        atm_params = {k: v for k, v in list(self.params.items())
                      if k in Model.ATM_KEYS}

        # TODO: Move scaling/PCA logic to method definition
        
        # if Model.PCA in atm_params and atm_params[Model.PCA]:
        #     whiten = (Model.WHITEN in atm_params and atm_params[Model.WHITEN])
        #     pca_dims = atm_params[Model.PCA_DIMS]
        #     # PCA dimension in atm_params is a float reprsenting percentages of
        #     # features to use
        #     if pca_dims < 1:
        #         dimensions = int(pca_dims * float(self.num_features))
        #         logger.info("Using PCA to reduce %d features to %d dimensions"
        #                     % (self.num_features, dimensions))
        #         pca = decomposition.PCA(n_components=dimensions, whiten=whiten)
        #         steps.append(('pca', pca))

        # should we scale the data?
        # if atm_params.get(Model.SCALE):
        #     steps.append(('standard_scale', StandardScaler()))
        # elif self.params.get(Model.MINMAX):
        #     steps.append(('minmax_scale', MinMaxScaler()))

        # add the classifier as the final step in the pipeline
        steps.append((self.method, self.classifier))
        self.pipeline = Pipeline(steps)

    def cross_validate(self, X, y):
        # TODO: this is hacky. See https://github.com/HDI-Project/ATM/issues/48
        binary = self.num_classes == 2
        kwargs = {}
        if self.verbose_metrics:
            kwargs['include_curves'] = True
            if not binary:
                kwargs['include_per_class'] = True

        df, cv_scores = cross_validate_pipeline(pipeline=self.pipeline,
                                                X=X, y=y, binary=binary,
                                                n_folds=self.N_FOLDS, **kwargs)

        self.cv_judgment_metric = np.mean(df[self.judgment_metric])
        self.cv_judgment_metric_stdev = np.std(df[self.judgment_metric])
        self.mu_sigma_judgment_metric = (self.cv_judgment_metric -
                                         2 * self.cv_judgment_metric_stdev)
        return cv_scores

    def test_final_model(self, X, y):
        """
        Test the (already trained) model pipeline on the provided test data
        (X and y). Store the test judgment metric and return the rest of the
        metrics as a hierarchical dictionary.
        """
        # time the prediction
        start_time = time.time()
        total = time.time() - start_time
        self.avg_predict_time = old_div(total, float(len(y)))

        # TODO: this is hacky. See https://github.com/HDI-Project/ATM/issues/48
        binary = self.num_classes == 2
        kwargs = {}
        if self.verbose_metrics:
            kwargs['include_curves'] = True
            if not binary:
                kwargs['include_per_class'] = True

        # compute the actual test scores!
        test_scores = test_pipeline(self.pipeline, X, y, binary, **kwargs)

        # save meta-metrics
        self.test_judgment_metric = test_scores.get(self.judgment_metric)

        return test_scores

    def train_test(self, X, y):
        # load train and (maybe) test data
        metadata = MetaData(X, y)
        self.num_classes = metadata.k_classes
        self.num_features = metadata.d_features

        # if necessary, cast judgment metric into its binary/multiary equivalent
        if self.num_classes == 2:
            if self.judgment_metric in [Metrics.F1_MICRO, Metrics.F1_MACRO]:
                self.judgment_metric = Metrics.F1
            elif self.judgment_metric in [Metrics.ROC_AUC_MICRO,
                                          Metrics.ROC_AUC_MACRO]:
                self.judgment_metric = Metrics.ROC_AUC
        else:
            if self.judgment_metric == Metrics.F1:
                self.judgment_metric = Metrics.F1_MACRO
            elif self.judgment_metric == Metrics.ROC_AUC:
                self.judgment_metric = Metrics.ROC_AUC_MACRO

        # generate train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=self.testing_ratio,
                                                     random_state=self.random_state)

        # We assume that all features & labels are numeric
        # create and cross-validate pipeline
        self._make_classifier()
        self.make_pipeline()
        cv_scores = self.cross_validate(X_train, y_train)

        # train and test the final model
        self.pipeline.fit(X_train, y_train)
        test_scores = self.test_final_model(X_test, y_test)
        return {'cv': cv_scores, 'test': test_scores}

    def save(self, model_dir, model_id):
        if issubclass(self.class_, methods.BaseMethod):
            return self.class_.Save(self.classifier, model_dir, model_id)
        else:
            return methods.BaseMethod.Save(self.classifier, model_dir, model_id)

    def load(self, model_dir, model_id):
        if issubclass(self.class_, methods.BaseMethod):
            self.classifier = self.class_.Load(model_dir, model_id, self.params)
        else:
            self.classifier = methods.BaseMethod.Load(model_dir, model_id, self.params)

        self.make_pipeline()  # Re-build pipeline with new classifier

    def destroy(self):
        destroy_func = getattr(self.classifier, 'destroy', None)
        if callable(destroy_func):
            destroy_func()

    def predict(self, X):
        """
        Use the pipeline to transform training data into
        predicted labels

        Args:
            X - n-d numpy array predict classes

        Returns: 1d numpy array of predictions 
        """
        predictions = self.pipeline.predict(X)

        return predictions

    def special_conversions(self, params):
        """
        TODO: replace this logic with something better
        """
        # create list parameters
        lists = defaultdict(list)
        element_regex = re.compile('(.*)\[(\d)\]')
        for name, param in list(params.items()):
            # look for variables of the form "param_name[1]"
            match = element_regex.match(name)
            if match:
                # name of the list parameter
                lname = match.groups()[0]
                # index of the list item
                index = int(match.groups()[1])
                lists[lname].append((index, param))

                # drop the element parameter from our list
                del params[name]

        for lname, items in list(lists.items()):
            # drop the list size parameter
            del params['len(%s)' % lname]

            # sort the list by index
            params[lname] = [val for idx, val in sorted(items)]

        # Gaussian process classifier
        if self.method == "gp":
            if params["kernel"] == "constant":
                params["kernel"] = ConstantKernel()
            elif params["kernel"] == "rbf":
                params["kernel"] = RBF()
            elif params["kernel"] == "matern":
                params["kernel"] = Matern(nu=params["nu"])
                del params["nu"]
            elif params["kernel"] == "rational_quadratic":
                params["kernel"] = RationalQuadratic(length_scale=params["length_scale"],
                                                     alpha=params["alpha"])
                del params["length_scale"]
                del params["alpha"]
            elif params["kernel"] == "exp_sine_squared":
                params["kernel"] = ExpSineSquared(length_scale=params["length_scale"],
                                                  periodicity=params["periodicity"])
                del params["length_scale"]
                del params["periodicity"]

        # return the updated parameter vector
        return params

    def _make_classifier(self):
        # create a classifier with specified parameters
        hyperparameters = {k: v for k, v in list(self.params.items())
                           if k not in Model.ATM_KEYS}
        # do special conversions
        hyperparameters = self.special_conversions(hyperparameters)
        self.classifier = self.class_(**hyperparameters)
