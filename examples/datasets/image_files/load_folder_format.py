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

import os
import re
import tempfile
import random
import csv
import shutil
from tqdm import tqdm
from PIL import Image
import argparse

def load(data_dir: str, image_width: int, image_height: int, 
        out_train_dataset_path: str, out_val_dataset_path: str, 
        out_meta_csv_path: str, validation_split: float):
    '''
        Loads and converts a set of labelled folders of images as a Rafiki-conformant dataset for IMAGE_CLASSIFICATION.

        :param str data_dir: Root directory containing the set of labelled folders
        :param int image_width: Width to resize all images to
        :param int image_height: Height to resize all images to
        :param str out_train_dataset_path: Path to save the output train dataset file
        :param str out_val_dataset_path: Path to save the output validation dataset file
        :param str out_meta_csv_path: Path to save the output dataset metadata .CSV file
        :param float validation_split: Proportion (0-1) to carve out validation dataset from the original dataset
    '''

    print('Loading datasets into memory...')
    (pil_images, labels, label_to_name) = _load_dataset_from_files(data_dir, image_width, image_height)
    (train_pil_images, train_labels, val_pil_images, val_labels) = _split_train_dataset(pil_images, labels, validation_split)

    print('Converting and writing datasets...')

    _write_meta_csv(label_to_name, out_meta_csv_path)
    print('Dataset metadata file is saved at {}'.format(out_meta_csv_path))

    _write_dataset(train_pil_images, train_labels, out_train_dataset_path)
    print('Train dataset file is saved at {}. This should be submitted as `train_dataset` of a train job.'
            .format(out_train_dataset_path))

    _write_dataset(val_pil_images, val_labels, out_val_dataset_path)
    print('Validation dataset file is saved at {}. This should be submitted as `val_dataset` of a train job.'
            .format(out_val_dataset_path))

def _split_train_dataset(train_pil_images, train_labels, validation_split):
    # Shuffle for consistency between train & val datasets
    zipped = list(zip(train_pil_images, train_labels))
    random.shuffle(zipped)
    (train_pil_images, train_labels) = zip(*zipped)

    val_start_idx = int(len(train_pil_images) * (1 - validation_split))
    val_pil_images = train_pil_images[val_start_idx:]
    val_labels = train_labels[val_start_idx:]
    train_pil_images = train_pil_images[:val_start_idx]
    train_labels = train_labels[:val_start_idx]
    return (train_pil_images, train_labels, val_pil_images, val_labels)

def _write_meta_csv(label_to_name, out_meta_csv_path):
    with open(out_meta_csv_path, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'name'])
        writer.writeheader()

        for (label, name) in label_to_name.items():
            writer.writerow({ 'class': label, 'name': name })

def _write_dataset(pil_images, labels, out_dataset_path):
    with tempfile.TemporaryDirectory() as d:
        # Create images.csv in temp dir for dataset
        # For each (image, label), save image as .png and add row to images.csv
        # Show a progress bar in the meantime
        images_csv_path = os.path.join(d, 'images.csv')
        n = len(pil_images)
        with open(images_csv_path, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'class'])
            writer.writeheader()
            for (i, pil_image, label) in tqdm(zip(range(n), pil_images, labels), total=n, unit='images'):
                image_name = '{}-{}.png'.format(label, i)
                image_path = os.path.join(d, image_name)
                pil_image.save(image_path)
                writer.writerow({ 'path': image_name, 'class': label })

        # Zip and export folder as dataset
        out_path = shutil.make_archive(out_dataset_path, 'zip', d)
        os.rename(out_path, out_dataset_path) # Remove additional trailing `.zip`

def _load_dataset_from_files(data_dir, image_width, image_height):
    sub_dir_names = get_immediate_subdir_names(data_dir)
    label_to_name = {}
    pil_images = []
    labels = []
    max_label = 0
    for label_name in sub_dir_names:
        image_names = get_image_names(os.path.join(data_dir, label_name))
        for name in image_names:
            pil_image = Image.open(os.path.join(data_dir, label_name, name))
            pil_image = pil_image.resize((image_width, image_height))
            pil_images.append(pil_image)
            labels.append(max_label)

        label_to_name[max_label] = label_name
        max_label += 1

    return (pil_images, labels, label_to_name)

def get_image_names(root_dir):
    return [x for x in os.listdir(root_dir) if re.search(r'\.(jpg|jpeg)$', x)]

def get_immediate_subdir_names(root_dir):
    return [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--image_width', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=32)
    parser.add_argument('--validation_split', type=float, default=0.1)
    args = parser.parse_args()

    load(
        data_dir=args.data_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        out_train_dataset_path='data/{}_train.zip'.format(args.name),
        out_val_dataset_path='data/{}_val.zip'.format(args.name),
        out_meta_csv_path='data/{}_meta.csv'.format(args.name),
        validation_split=args.validation_split,
    )

    