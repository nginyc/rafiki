#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from PIL import Image
import numpy as np
import random
import logging
from typing import List
import os
import traceback
import zipfile
import io
import abc
import csv
import pandas
import tempfile

logger = logging.getLogger(__name__)

class InvalidDatasetFormatException(Exception): pass

class DatasetUtils():
    '''
    Collection of utility methods to help with the loading of datasets.

    Usage of these methods are optional.
    In fact, you are encouraged to rely on your preferred ML libraries' dataset loading methods, or hand-roll your own.

    This should NOT be initiailized outside of the module. Instead,
    import the global ``utils`` instance from the module ``rafiki.model``
    and use ``utils.dataset``.

    For example:

    ::

        from rafiki.model import utils
        ...
        def train(self, dataset_path, **kwargs):
            ...
            utils.dataset.load_dataset_of_image_files(dataset_path)
            ...
    '''
    def load_dataset_of_corpus(self, dataset_path, tags=None, split_by='\\n'):
        '''
            Loads dataset for the task ``POS_TAGGING``

            :param str dataset_path: File path of the dataset
            :returns: An instance of ``CorpusDataset``.
        '''
        return CorpusDataset(dataset_path, tags or ['tag'], split_by)

    def load_dataset_of_image_files(self, dataset_path, min_image_size=None,
                                    max_image_size=None, mode='RGB', if_shuffle=False):
        '''
            Loads dataset for the task ``IMAGE_CLASSIFICATION``.

            :param str dataset_path: File path of the dataset
            :param int min_image_size: minimum width *and* height to resize all images to
            :param int max_image_size: maximum width *and* height to resize all images to
            :param str mode: Pillow image mode. Refer to https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
            :param bool if_shuffle: Whether to shuffle the dataset
            :returns: An instance of ``ImageFilesDataset``
        '''
        return ImageFilesDataset(dataset_path, min_image_size=min_image_size, max_image_size=max_image_size,
                                mode=mode, if_shuffle=if_shuffle)

    def load_dataset_of_audio_files(self, dataset_path, dataset_dir):
        '''
            Loads dataset with type `AUDIO_FILES`.

            :param str dataset_uri: URI of the dataset file
            :returns: An instance of ``AudioFilesDataset``.
        '''
        return AudioFilesDataset(dataset_path, dataset_dir)

    def normalize_images(self, images, mean=None, std=None):
        '''
            Normalize all images.

            If mean `mean` and standard deviation `std` are `None`, they will be computed on the images.

            :param images: (N x width x height x channels) array-like of images to resize
            :param float[] mean: Mean for normalization, by channel
            :param float[] std: Standard deviation for normalization, by channel
            :returns: (images, mean, std)
        '''
        if len(images) == 0:
            return (images, mean, std)

        # Convert to [0, 1]
        images = np.asarray(images) / 255

        if mean is None:
            mean = np.mean(images, axis=(0, 1, 2)).tolist() # shape = (channels,)
        if std is None:
            std = np.std(images, axis=(0, 1, 2)).tolist() # shape = (channels,)

        # Normalize all images
        images = (images - mean) / std

        return (images, mean, std)

    def transform_images(self, images, image_size=None, mode=None):
        '''
            Resize or convert a list of N images to another size and/or mode

            :param images: (N x width x height x channels) array-like of images to resize
            :param int image_size: width *and* height to resize all images to
            :param str mode: Pillow image mode to convert all images to. Refer to https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
            :returns: output images as a (N x width x height x channels) numpy array
        '''
        images = [Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in images]

        if image_size is not None:
            images = [x.resize([image_size, image_size]) for x in images]

        if mode is not None:
            images = [x.convert(mode) for x in images]

        return np.asarray([np.asarray(x) for x in images])

    def load_images(self, image_paths: List[str], mode: str = 'RGB') -> np.ndarray:
        '''
            Loads multiple images from the local filesystem.
            Assumes that images are of the same dimensions.

            :param images: Paths to images
            :param str mode: Pillow image mode to convert all images to. Refer to https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
            :returns: images as a (N x width x height x channels) numpy array
        '''
        pil_images = _load_pil_images(image_paths, mode=mode)
        images = np.array([np.asarray(x) for x in pil_images])
        return images

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
    Class that helps loading of dataset for task ``POS_TAGGING``.

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

            except:
                traceback.print_stack()
                raise InvalidDatasetFormatException()

        size = len(sents)

        return (size, tag_num_classes, max_token_len, max_sent_len, sents)

class ImageFilesDataset(ModelDataset):
    '''
    Class that helps loading of dataset for task ``IMAGE_CLASSIFICATION``.

    ``classes`` is the number of image classes.

    Each dataset example is (image, class) where:

        - Each image is a 3D numpy array (width x height x channels)
        - Each class is an integer from 0 to (k - 1)
    '''

    def __init__(self, dataset_path, min_image_size=None, max_image_size=None, mode='RGB', if_shuffle=False):
        super().__init__(dataset_path)
        (pil_images, self._image_classes, self.size, self.classes) = self._load(self.path, mode)
        (self._images, self.image_size) = self._preprocess(pil_images, min_image_size, max_image_size)
        if if_shuffle:
            (self._images, self._image_classes) = self._shuffle(self._images, self._image_classes)

    def __getitem__(self, index):
        image = self._images[index]
        image_class = self._image_classes[index]
        return (image, image_class)

    def _preprocess(self, pil_images, min_image_size, max_image_size):
        if len(pil_images) == 0:
            raise InvalidDatasetFormatException('Dataset should contain at least 1 image!')

        # Decide on image size, adhering to min/max, making it square and trying not to stretch it
        (width, height) = pil_images[0].size
        image_size = max(min([width, height, max_image_size or width]), min_image_size or 0)

        # Resize all images
        pil_images = [x.resize([image_size, image_size]) for x in pil_images]

        # Convert to numpy arrays
        images = [np.asarray(x) for x in pil_images]

        return (images, image_size)

    def _load(self, dataset_path, mode):
        # Create temp directory to unzip to
        with tempfile.TemporaryDirectory() as d:
            dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
            dataset_zipfile.extractall(path=d)
            dataset_zipfile.close()

            # Read images.csv, and read image paths & classes
            image_paths = []
            image_classes = []
            images_csv_path = os.path.join(d, 'images.csv')
            try:
                with open(images_csv_path, mode='r') as f:
                    reader = csv.DictReader(f)
                    (image_paths, image_classes) = zip(*[(row['path'], int(row['class'])) for row in reader])
            except:
                traceback.print_stack()
                raise InvalidDatasetFormatException()

            # Load images from files
            full_image_paths = [os.path.join(d, x) for x in image_paths]
            pil_images = _load_pil_images(full_image_paths, mode=mode)

        num_classes = len(set(image_classes))
        num_samples = len(image_paths)

        return (pil_images, image_classes, num_samples, num_classes)

    def _shuffle(self, images, classes):
        zipped = list(zip(images, classes))
        random.shuffle(zipped)
        (images, classes) = zip(*zipped)
        return (images, classes)

class AudioFilesDataset(ModelDataset):
    '''
    Class that helps loading of dataset with type `AUDIO_FILES`

    ``dataset_path`` is the URI to the dataset.
    ``dataset_dir`` is the directory to store the extracted the Audio Files.
    '''

    def __init__(self, dataset_path, dataset_dir):
        super().__init__(dataset_path)
        self._dataset_dir = dataset_dir
        self.df = self._load(self.path)

    def _load(self, dataset_path):
        '''
            Loading the dataset into a pandas dataframe. Called in the class __init__ method.

            :param str dataset_path: URI to the dataset
            :returns: pandas dataframe with three columns: ``wav_filename``, ``wav_filesize`` and ``transcript``
        '''
        dataset_dir = self._dataset_dir

        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        dataset_zipfile.extractall(path=dataset_dir.name)
        dataset_zipfile.close()

        # Read images.csv, and read image paths & classes
        audios_csv_path = os.path.join(dataset_dir.name, 'audios.csv')

        df = pandas.read_csv(audios_csv_path, encoding='utf-8', na_filter=False)

        return df
        
def _load_pil_images(image_paths, mode='RGB'):
    pil_images = []
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            encoded = io.BytesIO(f.read())
            pil_image = Image.open(encoded).convert(mode)
            pil_images.append(pil_image)
    
    return pil_images
