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

from examples.datasets.corpus.ptb import load

# Loads the Penn Treebank sample dataset for the `POS_TAGGING` task
def load_sample_ptb(out_train_dataset_path='data/ptb_train.zip',
                    out_val_dataset_path='data/ptb_val.zip',
                    out_meta_tsv_path='data/ptb_meta.tsv',
                    validation_split=0.05):
    load(
        dataset_url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip',
        out_train_dataset_path=out_train_dataset_path,
        out_val_dataset_path=out_val_dataset_path,
        out_meta_tsv_path=out_meta_tsv_path,
        validation_split=validation_split
    )

if __name__ == '__main__':
    load_sample_ptb()    
    