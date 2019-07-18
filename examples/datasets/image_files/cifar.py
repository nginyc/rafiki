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

import tarfile
import os
import tempfile
import pickle
import numpy as np
import csv
import shutil
from tqdm import tqdm
from itertools import chain
from PIL import Image

from examples.datasets.utils import download_dataset_from_url

def load(dataset_url, label_to_name, out_train_dataset_path, out_val_dataset_path, 
        out_test_dataset_path, out_meta_csv_path, validation_split, limit=None):
    '''
        Loads and converts an image dataset of the CIFAR format for IMAGE_CLASSIFICATION.
        Refer to https://www.cs.toronto.edu/~kriz/cifar.html for the CIFAR dataset format.

        :param str dataset_url: URL to download the Python version of the dataset
        :param dict[int, str] label_to_name: Dictionary mapping label index to label name
        :param str out_train_dataset_path: Path to save the output train dataset file
        :param str out_val_dataset_path: Path to save the output validation dataset file
        :param str out_test_dataset_path: Path to save the output test dataset file
        :param str out_meta_csv_path: Path to save the output dataset metadata .CSV file
        :param float validation_split: Proportion (0-1) to carve out validation dataset from the originl train dataset
        :param int limit: Maximum number of samples for each dataset (for purposes of development)
    '''
    if all([os.path.exists(x) for x in [out_train_dataset_path, out_val_dataset_path, out_meta_csv_path]]):
        print('Dataset already loaded in local filesystem - skipping...')
        return

    print('Downloading dataset archive...')
    dataset_zip_file_path = download_dataset_from_url(dataset_url)

    print('Loading datasets into memory...')
    (train_images, train_labels, test_images, test_labels) = _load_dataset_from_zip_file(dataset_zip_file_path)
    (train_images, train_labels, val_images, val_labels) = _split_train_dataset(train_images, train_labels, validation_split)

    print('Converting and writing datasets...')

    (label_to_index) = _write_meta_csv(chain(train_labels, test_labels), label_to_name, out_meta_csv_path)
    print('Dataset metadata file is saved at {}'.format(out_meta_csv_path))

    _write_dataset(train_images, train_labels, label_to_index, out_train_dataset_path, limit)
    print('Train dataset file is saved at {}. This should be submitted as `train_dataset` of a train job.'
            .format(out_train_dataset_path))

    _write_dataset(val_images, val_labels, label_to_index, out_val_dataset_path, limit)
    print('Validation dataset file is saved at {}. This should be submitted as `val_dataset` of a train job.'
            .format(out_val_dataset_path))

    _write_dataset(test_images, test_labels, label_to_index, out_test_dataset_path, limit)
    print('Test dataset file is saved at {}'.format(out_test_dataset_path))

def _split_train_dataset(train_images, train_labels, validation_split):
    val_start_idx = int(len(train_images) * (1 - validation_split))
    val_images = train_images[val_start_idx:]
    val_labels = train_labels[val_start_idx:]
    train_images = train_images[:val_start_idx]
    train_labels = train_labels[:val_start_idx]
    return (train_images, train_labels, val_images, val_labels)

def _write_meta_csv(labels, label_to_name, out_meta_csv_path):
    label_to_index = {}
    with open(out_meta_csv_path, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'name'])
        writer.writeheader()

        for (i, label) in enumerate(sorted(set(labels))):
            label_to_index[label] = i
            writer.writerow({ 'class': i, 'name': label_to_name[label] })

    return (label_to_index)

def _write_dataset(images, labels, label_to_index, out_dataset_path, limit):
    if limit is not None:
        print('Limiting dataset to {} samples...'.format(limit))
        images = images[:limit]
        labels = labels[:limit]

    with tempfile.TemporaryDirectory() as d:
        # Create images.csv in temp dir for dataset
        # For each (image, label), save image as .png and add row to images.csv
        # Show a progress bar in the meantime
        images_csv_path = os.path.join(d, 'images.csv')
        n = len(images)
        with open(images_csv_path, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'class'])
            writer.writeheader()
            for (i, image, label) in tqdm(zip(range(n), images, labels), total=n, unit='images'):
                image_name = '{}-{}.png'.format(label, i)
                image_path = os.path.join(d, image_name)
                pil_image = Image.fromarray(image, mode='RGB')
                pil_image.save(image_path)
                writer.writerow({ 'path': image_name, 'class': label_to_index[label] })

        # Zip and export folder as dataset
        out_path = shutil.make_archive(out_dataset_path, 'zip', d)
        os.rename(out_path, out_dataset_path) # Remove additional trailing `.zip`

def _load_dataset_from_zip_file(dataset_zip_file_path):
    with tempfile.TemporaryDirectory() as d:
        tar = tarfile.open(dataset_zip_file_path, 'r:gz')
        tar.extractall(path=d)
        tar.close()

        # Get root of data files
        files = os.listdir(d)
        if len(files) != 1:
            raise ValueError('Invalid format - root of archive should contain exactly 1 folder')
        root = os.path.join(d, files[0])

        # Extract train images & labels (merge batches)
        train_images = []
        train_labels = []
        for i in [1, 2, 3, 4, 5]:
            batch_file_path = os.path.join(root, 'data_batch_{}'.format(i))
            with open(batch_file_path, 'rb') as fo:
                train_data_batch = pickle.loads(fo.read(), encoding='bytes')
                (images, labels) = _cifar_batch_to_data(train_data_batch)
                train_images.extend(images)
                train_labels.extend(labels)

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        
        # Extract test images & labels
        test_file_path = os.path.join(root, 'test_batch')
        with open(test_file_path, 'rb') as fo:
            test_data = pickle.loads(fo.read(), encoding='bytes')
            (test_images, test_labels) = _cifar_batch_to_data(test_data)

    return (train_images, train_labels, test_images, test_labels)

def _cifar_batch_to_data(cifar_batch):
    images = _cifar_images_to_images(cifar_batch['data'.encode()])
    labels = cifar_batch['labels'.encode()]
    return (images, labels)

def _cifar_images_to_images(cifar_images):
    images = np.reshape(cifar_images, (-1, 3, 32 * 32))
    images = np.swapaxes(images, 1, 2)
    images = np.reshape(images, (-1, 32, 32, 3))
    return images

    
    