from flask import request
import os
import numpy as np

def get_request_params():
    # Get params from body as JSON
    params = request.get_json()

    # If the above fails, get params from body as form data
    if params is None:
        params = request.form.to_dict()

    # Merge in query params
    # We assume query params are always single-valued
    query_params = {
        k: v[0]
        for k, v in request.args.items()
    }
    params = {**params, **query_params}

    return params

def to_json_serializable(data):
    if isinstance(data, np.int64):
        return int(data)

    return data
