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

from sklearn import svm
import pickle
import base64
import numpy as np
import argparse

from rafiki.model import BaseModel, CategoricalKnob, FloatKnob, FixedKnob, utils
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

class SkSvm(BaseModel):
    '''
    Implements a SVM on Scikit-Learn for IMAGE_CLASSIFICATION
    '''
    @staticmethod
    def get_knob_config():
        return {
            'max_iter': FixedKnob(20),
            'kernel': CategoricalKnob(['rbf', 'linear', 'poly']),
            'gamma': CategoricalKnob(['scale', 'auto']),
            'C': FloatKnob(1e-4, 1e4, is_exp=True),
            'max_image_size': CategoricalKnob([16, 32])
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.max_iter, self.kernel, self.gamma, self.C)
        
    def train(self, dataset_path, **kwargs):
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=self.max_image_size, mode='L')
        self._image_size = dataset.image_size
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        X = self._prepare_X(images)
        y = classes
        self._clf.fit(X, y)

    def evaluate(self, dataset_path):
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=self.max_image_size, mode='L')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        X = self._prepare_X(images)
        y = classes
        preds = self._clf.predict(X)
        accuracy = sum(y == preds) / len(y)
        return accuracy

    def predict(self, queries):
        queries = utils.dataset.transform_images(queries, image_size=self._image_size, mode='L')
        X = self._prepare_X(queries)
        probs = self._clf.predict_proba(X)
        return probs.tolist()

    def dump_parameters(self):
        params = {}

        # Save model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64

        # Save image size
        params['image_size'] = self._image_size
        
        return params

    def load_parameters(self, params):
        # Load model parameters
        clf_base64 = params['clf_base64']

        # Load image size
        self._image_size = params['image_size']
        
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)

    def _prepare_X(self, images):
        return [np.asarray(image).flatten() for image in images]

    def _build_classifier(self, max_iter, kernel, gamma, C):
        clf = svm.SVC(
            max_iter=max_iter,
            kernel=kernel,
            gamma=gamma,
            C=C,
            probability=True
        ) 
        return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/fashion_mnist_train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/fashion_mnist_val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/fashion_mnist_test.zip', help='Path to test dataset')
    parser.add_argument('--query_path', type=str, default='examples/data/image_classification/fashion_mnist_test_1.png', 
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    test_model_class(
        model_file_path=__file__,
        model_class='SkSvm',
        task='IMAGE_CLASSIFICATION',
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )
