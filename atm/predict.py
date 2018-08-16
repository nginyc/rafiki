from __future__ import absolute_import, division, unicode_literals

import logging
import os
from builtins import map
from datetime import datetime, timedelta
import numpy as np

from .model import Model

from prepare import create_preparator

# load the library-wide logger
logger = logging.getLogger('atm')


def predict(db, classifier_id, queries, log_config):
    """
    Make batch predictions with the trained classifier

    Args:
        db - initialized Database object
        classifier_id: ID of classifier to query
        queries: raw input data compatible with the configured preparator for the datarun
        log_config: log config for the prediction 

    Returns: list of predictions

    """

    clf = db.get_classifier(classifier_id)
    
    if not clf:
        raise Exception('No such classifier')

    # Load model from DB & disk
    hp = db.get_hyperpartition(clf.hyperpartition_id)
    dr = db.get_datarun(clf.datarun_id)
    ds = db.get_dataset(dr.dataset_id)
    model = Model(method=hp.method, 
                  params=clf.hyperparameter_values,
                  num_classes=ds.k_classes,
                  judgment_metric=dr.metric)
    model.load(log_config.model_dir, clf.id)

    # Create data preparator
    preparator = create_preparator(ds.preparator_type, **ds.preparator_params)

    # Extract X from queries
    X, _ = preparator.transform_data(queries)

    # Make predictions
    y = model.predict(X)

    _, predictions = preparator.reverse_transform_data(y=y)

    # Clean up model
    model.destroy()

    return predictions

def get_dataset_example(db, dataset_id, example_id=None):
    ds = db.get_dataset(dataset_id)

    if not ds:
        raise Exception('No such dataset')

    # Create data preparator
    preparator = create_preparator(ds.preparator_type, **ds.preparator_params)

    x, y = preparator.get_train_example(example_id)
    
    queries, labels = preparator.reverse_transform_data(
        X=np.array([x]), 
        y=np.array([y])
    )
    
    return queries[0], labels[0]

    