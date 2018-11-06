from PIL import Image
import numpy as np
import requests
import logging
import os
import tempfile
import traceback
from tqdm import tqdm
import math
from urllib.parse import urlparse
import zipfile
import io
import tempfile
import csv

from rafiki.constants import DatasetType

logger = logging.getLogger(__name__)

class InvalidDatasetProtocolException(Exception): pass 
class InvalidDatasetTypeException(Exception): pass 
class InvalidDatasetFormatException(Exception): pass 

class ModelDatasetUtils():
    def __init__(self):
        # Caches downloaded datasets
        self._dataset_uri_to_path = {}
        
    def load_dataset_of_image_files(self, dataset_uri):
        '''
            Returns a generator for the dataset of type `IMAGE_FILES`.
            
            The first yield gives (<number of examples>, <number of classes>).
            Subsequent yields give (image, class) where each image is a 2D numpy array 
                of integers (0, 255) as grayscale, each class is an integer from 0 to (k - 1).
        '''
        dataset_path = self.download_dataset_from_uri(dataset_uri)
        
        with tempfile.TemporaryDirectory() as d:
            dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
            dataset_zipfile.extractall(path=d)
            dataset_zipfile.close()

            # Read images.csv, yield metadata, then yield sample by sample
            images_csv_path = os.path.join(d, 'images.csv') 
            try:
                with open(images_csv_path, mode='r') as f:
                    reader = csv.DictReader(f)
                    (image_paths, image_classes) = zip(*[(row['path'], int(row['class'])) for row in reader])
                    num_classes = len(set(image_classes))
                    num_samples = len(image_paths)

                    yield (num_samples, num_classes)

                    for (image_path, image_class) in zip(image_paths, image_classes):
                        full_image_path = os.path.join(d, image_path)
                        with open(full_image_path, 'rb') as f:
                            encoded = io.BytesIO(f.read())
                            image = np.array(Image.open(encoded))

                        yield (image, image_class)

            except Exception:
                traceback.print_stack()
                raise InvalidDatasetFormatException()
                
    def download_dataset_from_uri(self, dataset_uri):
        '''
            Maybe download the dataset at URI, ensuring that the dataset ends up in the local filesystem.

            :param str dataset_uri: URI of the dataset file
            :returns: file path of the dataset file in the local filesystem
        '''
        if dataset_uri in self._dataset_uri_to_path:
            return self._dataset_uri_to_path[dataset_uri]

        dataset_path = None

        parsed_uri = urlparse(dataset_uri)
        protocol = '{uri.scheme}'.format(uri=parsed_uri).lower().strip()

        # Download dataset over HTTP/HTTPS
        if protocol == 'http' or protocol == 'https':

            r = requests.get(dataset_uri, stream=True)
            temp_file = tempfile.NamedTemporaryFile(delete=False)

            # Show a progress bar while downloading
            total_size = int(r.headers.get('content-length', 0)); 
            block_size = 1024
            iters = math.ceil(total_size / block_size) 
            for data in tqdm(r.iter_content(block_size), total=iters, unit='KB'):
                temp_file.write(data)
                
            temp_file.close()
            
            dataset_path = temp_file.name

        # Assume it is on filesystem
        elif protocol == '' or protocol == 'file':

            dataset_path = dataset_uri

        else:
            raise InvalidDatasetProtocolException()

        # Cache dataset path to possibly prevent re-downloading
        self._dataset_uri_to_path[dataset_uri] = dataset_path
        return dataset_path


