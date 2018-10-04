from PIL import Image
import numpy as np
import requests

from urllib.parse import urlparse
import zipfile
import io

from rafiki.constants import *

def load_dataset(uri, task):
    parsed_uri = urlparse(uri)
    protocol = '{uri.scheme}'.format(uri=parsed_uri)
    if _is_http(protocol) or _is_https(protocol):
        if _is_image_classification(task):
            r = requests.get(uri)
            images = []
            labels = []
            with zipfile.ZipFile(io.BytesIO(r.content)) as dataset:
                for entry in dataset.namelist():
                    if entry.endswith('.png') or entry.endswith('.jpg'):
                        label = entry.split('/')[-2]
                        labels.append(label)
                        encoded = io.BytesIO(dataset.read(entry))
                        image = np.array(Image.open(encoded))
                        images.append(image)
            return (np.array(images), np.array(labels))
        else:
            raise Exception('{} task not supported'.format(task))
    else:
        raise Exception('Dataset URI scheme not supported: {}'.format(protocol))

def _is_http(protocol):
    return protocol == DatasetProtocol.HTTP

def _is_https(protocol):
    return protocol == DatasetProtocol.HTTPS

def _is_image_classification(task):
    return task == DatasetTask.IMAGE_CLASSIFICATION
