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
import abc
import tempfile
import csv
import pandas as pd

from rafiki.constants import DatasetType

logger = logging.getLogger(__name__)

class InvalidDatasetProtocolException(Exception): pass 
class InvalidDatasetTypeException(Exception): pass 
class InvalidDatasetFormatException(Exception): pass 

class ModelDatasetUtils():
    '''
    Collection of utility methods to help with the loading of datasets.

    To use these utility methods, import the global ``dataset_utils`` instance from the module ``rafiki.model``.

    For example:

    ::

        from rafiki.model import dataset_utils
        ...
        def train(self, dataset_uri):
            ...
            dataset_utils.load_dataset_of_image_files(dataset_uri)
            ...
    '''   
    
    def __init__(self):
        # Caches downloaded datasets
        self._dataset_uri_to_path = {}

    def load_dataset_of_corpus(self, dataset_uri, tags=['tag'], split_by='\\n'):
        '''
            Loads dataset with type `CORPUS`.
            
            :param str dataset_uri: URI of the dataset file
            :returns: An instance of ``CorpusDataset``.
        '''
        dataset_path = self.download_dataset_from_uri(dataset_uri)
        return CorpusDataset(dataset_path, tags, split_by)

    def load_dataset_of_image_files(self, dataset_uri, image_size=None):
        '''
            Loads dataset with type `IMAGE_FILES`.

            :param str dataset_uri: URI of the dataset file
            :param str image_size: dimensions to resize all images to (None for no resizing)
            :returns: An instance of ``ImageFilesDataset``.
        '''
        dataset_path = self.download_dataset_from_uri(dataset_uri)
        return ImageFilesDataset(dataset_path, image_size)

    def load_dataset_of_tabular(self, dataset_path):

        """
            Loads dataset for the task ``TabularRegression`` or ``TabularClassification``.

            :param str dataset_path: File path of the dataset
            :returns: An instance of ``TabularDataset``
        """
        return TabularDataset(dataset_path)

    def resize_as_images(self, images, image_size):
        '''
            Resize a list of N grayscale images to another size.

            :param int[][][] images: images to resize as a N x 2D lists (grayscale)
            :param int image_size: dimensions to resize all images to (None for no resizing)
            :returns: images as N x 2D numpy arrays
        '''
        images = [Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in images]
        images = [np.asarray(x.resize(image_size)) for x in images]
        return np.asarray(images)
                
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

class ModelDataset():
    '''
    Abstract that helps loading of dataset of a specific type
    
    ``size`` should be the total number of samples of the dataset
    '''
    def __init__(self, dataset_path):
        super().__init__()
        self.path = dataset_path
        self.size = 0

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return self.size

class CorpusDataset(ModelDataset):
    '''
    Class that helps loading of dataset with type `CORPUS`

    ``tags`` is the expected list of tags for each token in the corpus.
    Dataset samples are grouped as sentences by a delimiter token corresponding to ``split_by``.
    
    ``tag_num_classes`` is a list of <number of classes for a tag>, in the same order as ``tags``.
    Each dataset sample is [[token, <tag_1>, <tag_2>, ..., <tag_k>]] where each token is a string, 
    each ``tag_i`` is an integer from 0 to (k_i - 1) as each token's corresponding class for that tag, 
    with tags appearing in the same order as ``tags``. 
    '''   

    def __init__(self, dataset_path, tags, split_by):
        super().__init__(dataset_path)
        self.tags = tags
        (self.size, self.tag_num_classes, self.max_token_len, self.max_sent_len, self._sents) = \
            self._load(self.path, self.tags, split_by)

    def __getitem__(self, index):
        return self._sents[index]

    def _load(self, dataset_path, tags, split_by):
        sents = []
        tag_num_classes = [0 for _ in range(len(tags))]
        max_token_len = 0
        max_sent_len = 0
        
        with tempfile.TemporaryDirectory() as d:
            dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
            dataset_zipfile.extractall(path=d)
            dataset_zipfile.close()

            # Read corpus.tsv, read token by token, and merge them into sentences
            corpus_tsv_path = os.path.join(d, 'corpus.tsv') 
            try:
                with open(corpus_tsv_path, mode='r') as f:
                    reader = csv.DictReader(f, dialect='excel-tab')

                    # Read full corpus into memory token by token
                    sent = []
                    for row in reader:
                        token = row['token']
                        del row['token']

                        # Start new sentence upon encountering delimiter
                        if token == split_by:
                            sents.append(sent)
                            sent = []
                            continue
                        
                        token_tags = [int(row[x]) for x in tags]
                        sent.append([token, *token_tags])

                        # Maintain max classes of tags
                        tag_num_classes = [max(x + 1, m) for (x, m) in zip(token_tags, tag_num_classes)]

                        # Maintain max token length
                        max_token_len = max(len(token), max_token_len)

                    # Maintain max sent length
                    max_sent_len = max(len(sent), max_sent_len)

            except Exception:
                traceback.print_stack()
                raise InvalidDatasetFormatException()

        size = len(sents)

        return (size, tag_num_classes, max_token_len, max_sent_len, sents)

class ImageFilesDataset(ModelDataset):
    '''
    Class that helps loading of dataset with type `IMAGE_FILES`
    
    ``classes`` is the number of image classes.
    Each dataset example is (image, class) where each image is a 2D list
    of integers (0, 255) as grayscale, each class is an integer from 0 to (k - 1).
    '''   

    def __init__(self, dataset_path, image_size):
        super().__init__(dataset_path)
        self.image_size = image_size
        (self.size, self.classes, self._image_paths, 
            self._image_classes, self._dataset_dir) = self._load(self.path)

    def __getitem__(self, index):
        image_path = self._image_paths[index]
        image_class = self._image_classes[index]
        dataset_dir = self._dataset_dir
        image_size = self.image_size

        full_image_path = os.path.join(dataset_dir.name, image_path)
        with open(full_image_path, 'rb') as f:
            encoded = io.BytesIO(f.read())
            image = Image.open(encoded)
            
            if image_size is not None:
                image = image.resize(image_size)

            image = np.asarray(image)

        return (image, image_class)

    def _load(self, dataset_path):
        image_paths = []
        image_classes = [] 

        # Create temp directory to unzip to
        dataset_dir = tempfile.TemporaryDirectory()

        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        dataset_zipfile.extractall(path=dataset_dir.name)
        dataset_zipfile.close()

        # Read images.csv, and read image paths & classes
        images_csv_path = os.path.join(dataset_dir.name, 'images.csv') 
        try:
            with open(images_csv_path, mode='r') as f:
                reader = csv.DictReader(f)
                (image_paths, image_classes) = zip(*[(row['path'], int(row['class'])) for row in reader])
        except Exception:
            traceback.print_stack()
            raise InvalidDatasetFormatException()

        num_classes = len(set(image_classes))
        num_samples = len(image_paths)

        return (num_samples, num_classes, image_paths, image_classes, dataset_dir)

class TabularDataset(ModelDataset):
    '''
    Class that helps loading of tabular format dataset``.

    Each dataset example is a csv file which will be loaded in to pandas DataFrame format
    '''   

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        (self.data, self.size) = self._load(self.path)
        
    def __getitem__(self, index):
        return self.data.iloc[:1,:]

    def _load(self, dataset_path):
        data = pd.read_csv(dataset_path)
        size = data.size
        return (data, size)

dataset_utils = ModelDatasetUtils()