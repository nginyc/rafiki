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
from examples.datasets.image_files.mnist import load

# Loads the official Fashion MNIST dataset for the `IMAGE_CLASSIFICATION` task
def load_fashion_mnist(out_train_dataset_path='data/fashion_mnist_train.zip',
                        out_val_dataset_path='data/fashion_mnist_val.zip',
                        out_meta_csv_path='data/fashion_mnist_meta.csv',
                        out_test_dataset_path='data/fashion_mnist_test.zip',
                        limit=None,
                        validation_split=0.1):
    
    load(
        train_images_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        train_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        test_images_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        test_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        label_to_name={
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        },
        out_train_dataset_path=out_train_dataset_path,
        out_val_dataset_path=out_val_dataset_path,
        out_test_dataset_path=out_test_dataset_path,
        out_meta_csv_path=out_meta_csv_path,
        limit=limit,
        validation_split=validation_split
    )

if __name__ == '__main__':   
    # Read CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--validation_split', type=float, default=0.1)
    args = parser.parse_args()

    load_fashion_mnist(limit=args.limit, validation_split=args.validation_split)    