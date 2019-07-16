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

import argparse

from examples.datasets.image_files.cifar import load

# Loads the official CIFAR-10 dataset for `IMAGE_CLASSIFICATION` task
def load_cifar10(out_train_dataset_path='data/cifar10_train.zip',
                out_val_dataset_path='data/cifar10_val.zip',
                out_test_dataset_path='data/cifar10_test.zip',
                out_meta_csv_path='data/cifar10_meta.csv',
                validation_split=0.1,
                limit=None):
    load(
        dataset_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        label_to_name={
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        },
        out_train_dataset_path=out_train_dataset_path,
        out_val_dataset_path=out_val_dataset_path,
        out_test_dataset_path=out_test_dataset_path,
        out_meta_csv_path=out_meta_csv_path,
        validation_split=validation_split,
        limit=limit
    )

if __name__ == '__main__':
    # Read CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--validation_split', type=float, default=0.1)
    args = parser.parse_args()

    load_cifar10(limit=args.limit, validation_split=args.validation_split)    
    