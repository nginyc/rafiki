from PIL import Image
import numpy as np
import requests
import logging
import os
import tempfile
import traceback

from urllib.parse import urlparse
import zipfile
import io

from rafiki.constants import TaskType, DatasetType

logger = logging.getLogger(__name__)

TASK_TYPE_TO_DATASET_TYPE = {
    TaskType.IMAGE_CLASSIFICATION: DatasetType.IMAGE_CLASSES_BY_FOLDERS
}

class InvalidTaskException(Exception): pass
class InvalidDatasetProtocolException(Exception): pass 
class InvalidDatasetTypeException(Exception): pass 
class InvalidDatasetFormatException(Exception): pass 

class ModelDatasetUtils():
    def load_dataset(self, dataset_uri, task):
        if task not in TASK_TYPE_TO_DATASET_TYPE:
            raise InvalidTaskException()

        dataset_type = TASK_TYPE_TO_DATASET_TYPE[task]
        (dataset, dataset_meta) = self.load_dataset_of_type(dataset_uri, dataset_type)
        return (dataset, dataset_meta)
        
    def load_dataset_of_type(self, dataset_uri, dataset_type):
        dataset_file = self.open_dataset_from_uri(dataset_uri)

        if dataset_type == DatasetType.IMAGE_CLASSES_BY_FOLDERS:
            (dataset, dataset_meta) = self.load_dataset_of_image_classes_by_folders(dataset_file)
        else:
            raise InvalidDatasetTypeException()

        os.remove(dataset_file.name)
        return (dataset, dataset_meta)

    def load_dataset_of_image_classes_by_folders(self, dataset_file):
        images = []
        labels = []

        try:
            with zipfile.ZipFile(dataset_file.name) as dataset:
                for entry in dataset.namelist():
                    if entry.endswith('.png') or entry.endswith('.jpg') or entry.endswith('.jpeg'):
                        label = entry.split('/')[-2]
                        labels.append(label)
                        encoded = io.BytesIO(dataset.read(entry))
                        image = np.array(Image.open(encoded))
                        images.append(image)
        except:
            traceback.print_stack()
            raise InvalidDatasetFormatException()

        class_names = np.unique(labels)
        num_classes = len(class_names)
        index_to_label = dict(zip(range(num_classes), class_names))
        label_to_index = {v: k for k, v in index_to_label.items()}
        labels = [label_to_index[label] for label in labels]

        return ((images, labels), index_to_label)

    def relabel_dataset_labels(self, labels, from_index_to_label, to_index_to_label):
        to_label_to_index = { v: k for (k, v) in to_index_to_label.items() }

        for i in labels:
            labels[i] = to_label_to_index[from_index_to_label[labels[i]]]
        
        return labels

    def open_dataset_from_uri(self, dataset_uri):
        parsed_uri = urlparse(dataset_uri)
        protocol = '{uri.scheme}'.format(uri=parsed_uri).lower()

        # Download dataset over HTTP/HTTPS if required
        if protocol == 'http' or protocol == 'https':
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            r = requests.get(dataset_uri)
            temp_file.write(r.content)
            temp_file.close()
            return temp_file
        else:
            raise InvalidDatasetProtocolException()
