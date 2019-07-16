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
import glob
import traceback
import shutil
from tqdm import tqdm
import zipfile

from examples.datasets.utils import download_dataset_from_url

def load(dataset_url, out_train_dataset_path, out_val_dataset_path, out_meta_tsv_path, validation_split):
    '''
        Loads and converts a dataset of the format of the Penn Treebank sample 
        at http://www.nltk.org/nltk_data/ to the dataset type `CORPUS` for the task `POS_TAGGING`.

        :param str dataset_url: URL to download the dataset stored in the format similar to the Penn Treebank sample
        :param str out_train_dataset_path: Path to save the output train dataset file
        :param str out_val_dataset_path: Path to save the output validation dataset file
        :param str out_meta_tsv_path: Path to save the output dataset metadata .TSV file
        :param float validation_split: Proportion (0-1) to carve out validation dataset from the original dataset
    '''
    if all([os.path.exists(x) for x in [out_train_dataset_path, out_val_dataset_path, out_meta_tsv_path]]):
        print('Dataset already loaded in local filesystem - skipping...')
        return

    dataset_path = download_dataset_from_url(dataset_url)

    print('Loading dataset and writing to output dataset files...')
    _convert_dataset(dataset_path, out_meta_tsv_path, out_train_dataset_path, 
                    out_val_dataset_path, validation_split)

    print('Dataset metadata file is saved at {}'.format(out_meta_tsv_path))
    print('Train dataset file is saved at {}. This should be submitted as `train_dataset` of a train job.'
            .format(out_train_dataset_path))
    print('Validation dataset file is saved at {}. This should be submitted as `val_dataset` of a train job.'
            .format(out_val_dataset_path))

def _convert_dataset(dataset_path, out_meta_tsv_path, out_train_dataset_path, 
                    out_val_dataset_path, validation_split):
    TAGGED_DIRNAME = 'treebank/tagged'
    SENTS_FILENAME_GLOB = '*.pos'
    TSV_FILENAME = 'corpus.tsv'

    # Create train dataset dir & start TSV
    train_d = tempfile.TemporaryDirectory()
    train_tsv = open(os.path.join(train_d.name, TSV_FILENAME), 'w')
    train_tsv.write('token\ttag\n') 

    # Same for validation dataset
    val_d = tempfile.TemporaryDirectory()
    val_tsv = open(os.path.join(val_d.name, TSV_FILENAME), 'w')
    val_tsv.write('token\ttag\n')

    tag_to_index = {}

    with tempfile.TemporaryDirectory() as d:
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        dataset_zipfile.extractall(path=d)
        dataset_zipfile.close()

        tagged_dirpath = os.path.join(d, TAGGED_DIRNAME)
        sents_filepaths = glob.glob(os.path.join(tagged_dirpath, SENTS_FILENAME_GLOB))
        sents_filepaths.sort()

        # Compute no. of sents files for train
        train_files_count = round(len(sents_filepaths) * (1 - validation_split))
        
        # Convert sentences for train dataset
        for sents_filepath in tqdm(sents_filepaths[0:train_files_count], unit='files'):
            with open(sents_filepath) as f:
                while True:
                    sent = _read_next_sentence(f, tag_to_index)
                    if len(sent) == 0: break
                    _write_next_sentence(train_tsv, sent)

        # Convert sentences for validation dataset
        for sents_filepath in tqdm(sents_filepaths[train_files_count:], unit='files'):
            with open(sents_filepath) as f:
                while True:
                    sent = _read_next_sentence(f, tag_to_index)
                    if len(sent) == 0: break
                    _write_next_sentence(val_tsv, sent)

    # Zip train & validation datasets
    val_tsv.close()
    train_tsv.close()
    out_path = shutil.make_archive(out_train_dataset_path, 'zip', train_d.name)
    os.rename(out_path, out_train_dataset_path) # Remove additional trailing `.zip`
    out_path = shutil.make_archive(out_val_dataset_path, 'zip', val_d.name)
    os.rename(out_path, out_val_dataset_path) # Remove additional trailing `.zip`

    # Write to out meta file
    index_to_tag = { v: k for (k, v) in tag_to_index.items() }
    with open(out_meta_tsv_path, mode='w') as f:
        f.write('tag\tname\n') # Header
        for i in range(len(index_to_tag)):
            f.write('{}\t{}\n'.format(i, index_to_tag[i]))
    
def _write_next_sentence(f, sent):
    for [token, tag] in sent:
        f.write('{}\t{}\n'.format(token, tag))

    # End off with new line
    f.write('\\n\t\n')

def _read_next_sentence(f, tag_to_index):
    # For each subsequent line, parse tokens on line and insert into sentence
    sent = []
    started = False
    while True:
        line = next(f, None)
        
        # If EOF
        if line is None:
            return sent

        # Ignore first few blank lines, or if no more lines for sentence left
        if re.match('^[=\s]+$', line):
            if started: 
                return sent
            else: 
                continue

        started = True

        try:
            line_tokens = re.findall('\S+/\S+', line)
            for line_token in line_tokens:
                match = re.match('(\S+)/(\S+)', line_token)
                token = match.group(1)
                tag = match.group(2)

                # Convert tags to indices
                if tag not in tag_to_index:
                    tag_to_index[tag] = len(tag_to_index)

                sent.append([token, tag_to_index[tag]])
        except:
            print('WARNING: Failed to parse line "{}"'.format(line))
            traceback.print_stack()
        