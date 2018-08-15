from __future__ import absolute_import, division, unicode_literals

import logging
import os
from builtins import map
from datetime import datetime, timedelta

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

    # Load model from DB & disk
    clf = db.get_classifier(classifier_id)
    hp = db.get_hyperpartition(clf.hyperpartition_id)
    dr = db.get_datarun(clf.datarun_id)
    ds = db.get_dataset(dr.dataset_id)
    model = Model(method=hp.method, params=clf.hyperparameter_values,
                  judgment_metric=dr.metric)
    model.load(log_config.model_dir, clf.id)

    # Create data preparator
    preparator = create_preparator(ds.preparator_type, **ds.preparator_params)

    # Extract X from queries data
    X, _ = preparator.process_data(queries)

    # Make predictions
    predictions = model.predict(X)

    # Clean up model
    model.destroy()

    # Dumb down data types for predictions
    predictions = [int(x) for x in predictions] 

    return predictions

