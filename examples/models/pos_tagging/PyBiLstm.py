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
import math
import sys
import random
import datetime
import numpy as np
import traceback
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        IntegerKnob, FloatKnob, CategoricalKnob, logger, dataset_utils
from rafiki.constants import TaskType, ModelDependency
from rafiki.config import APP_MODE

class PyBiLstm(BaseModel):
    '''
    Implements a Bidrectional LSTM model in Pytorch for POS tagging
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': IntegerKnob(10, 50 if APP_MODE != 'DEV' else 10),
            'word_embed_dims': IntegerKnob(16, 128),
            'word_rnn_hidden_size': IntegerKnob(16, 128),
            'word_dropout': FloatKnob(1e-3, 2e-1, is_exp=True),
            'learning_rate': FloatKnob(1e-2, 1e-1, is_exp=True),
            'batch_size': CategoricalKnob([16, 32, 64, 128]),
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs

    def train(self, dataset_uri):
        dataset = dataset_utils.load_dataset_of_corpus(dataset_uri)
        self._word_dict = self._extract_word_dict(dataset)
        self._tag_count = dataset.tag_num_classes[0] 

        logger.log('No. of unique words: {}'.format(len(self._word_dict)))
        logger.log('No. of tags: {}'.format(self._tag_count))
        
        (self._net, self._optimizer) = self._train(dataset)
        sents_tags = self._predict(dataset)
        acc = self._compute_accuracy(dataset, sents_tags)

        logger.log('Train accuracy: {}'.format(acc))

    def evaluate(self, dataset_uri):
        dataset = dataset_utils.load_dataset_of_corpus(dataset_uri)
        sents_tags = self._predict(dataset)
        acc = self._compute_accuracy(dataset, sents_tags)
        return acc

    def predict(self, queries):
        queries = [[[x] for x in y] for y in queries]
        sents_tags = self._predict(queries)
        return sents_tags

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}
        params['net_state_dict'] = self._net.state_dict()
        params['optimizer_state_dict'] = self._optimizer.state_dict()
        params['word_dict'] = self._word_dict
        params['tag_count'] = self._tag_count
        return params

    def load_parameters(self, params):
        self._word_dict = params['word_dict']
        self._tag_count = params['tag_count']
        (net, optimizer) = self._create_model()
        net_state_dict = params['net_state_dict']
        net.load_state_dict(net_state_dict)
        optimizer_state_dict = params['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state_dict)

        self._net = net
        self._optimizer = optimizer

    def _extract_word_dict(self, dataset):
        word_dict = {}

        for sent in dataset:
            for [word, tag] in sent:
                if word not in word_dict:
                    word_dict[word] = len(word_dict)

        return word_dict    

    def _prepare_batch(self, dataset, lo, hi, Tensor, has_tags=True):
        word_dict = self._word_dict
        word_count = len(self._word_dict) 
        null_word = word_count
        null_tag = self._tag_count

        batch = dataset[lo:hi]
        N = len(batch)
        W = max([len(x) for x in batch]) # Max sent length

        # Prepare words
        words_tsr = Tensor(N, W)
        for i in range(N):
            for w in range(W):
                if w < len(batch[i]):
                    word = batch[i][w][0]
                    word = word_dict.get(word, random.randint(0, word_count - 1))
                    words_tsr[i][w] = word
                else:
                    words_tsr[i][w] = null_word

        # Prepare tags
        tags_tsr = None
        if has_tags:
            tags_tsr = Tensor(N, W)
            for i in range(N):
                for w in range(W):
                    if w < len(batch[i]):
                        tag = batch[i][w][1]
                        tags_tsr[i][w] = tag
                    else:
                        tags_tsr[i][w] = null_tag

        return (words_tsr, tags_tsr)

    def _predict(self, dataset):
        N = self._knobs.get('batch_size')
        net = self._net
        B = math.ceil(len(dataset) / N) # No. of batches
        word_count = len(self._word_dict)
        null_word = word_count

        Tensor = torch.LongTensor
        if torch.cuda.is_available():
            logger.log('Using CUDA...')
            net = net.cuda()
            Tensor = torch.cuda.LongTensor

        sents_pred_tags = []
        for i in range(B):
            # Extract batch from dataset 
            (words_tsr, _) = self._prepare_batch(dataset, i * N, i * N + N, Tensor, has_tags=False)

            # Forward propagate batch through model
            probs_tsr = net(words_tsr)

            # Compute sum of per-word loss for all words & sentences
            _, preds_tsr = torch.max(probs_tsr, dim=2)

            # Populate predictions
            for (sent_preds_tsr, sent_words_tsr) in zip(preds_tsr, words_tsr):
                sent_pred_tags = []

                for (pred, word) in zip(sent_preds_tsr, sent_words_tsr):
                    if word.item() == null_word: break
                    sent_pred_tags.append(pred.item())

                sents_pred_tags.append(sent_pred_tags)

        return sents_pred_tags

    def _train(self, dataset):
        N = self._knobs.get('batch_size')
        ep = self._knobs.get('epochs')
        null_tag = self._tag_count # Tag to ignore (from padding of sentences during batching)
        B = math.ceil(len(dataset) / N) # No. of batches

        # Define 2 plots: Loss against time, loss against epochs
        logger.define_loss_plot()
        logger.define_plot('Loss Over Time', ['loss'])

        (net, optimizer) = self._create_model()

        Tensor = torch.LongTensor
        if torch.cuda.is_available():
            logger.log('Using CUDA...')
            net = net.cuda()
            Tensor = torch.cuda.LongTensor

        loss_func = nn.CrossEntropyLoss(ignore_index=null_tag)

        for epoch in range(ep):
            total_loss = 0
            for i in range(B):
                # Extract batch from dataset 
                (words_tsr, tags_tsr) = self._prepare_batch(dataset, i * N, i * N + N, Tensor)

                # Reset gradients for this batch
                optimizer.zero_grad()

                # Forward propagate batch through model
                probs_tsr = net(words_tsr)

                # Compute sum of per-word loss for all words & sentences
                NW = probs_tsr.size(0) * probs_tsr.size(1)
                loss = loss_func(probs_tsr.view(NW, -1), tags_tsr.view(-1))
                
                # Backward propagate on minibatch
                loss.backward()

                # Update gradients with optimizer
                optimizer.step()

                total_loss += loss.item()

            logger.log_loss(loss=(total_loss / B), epoch=epoch)

        return (net, optimizer)

    def _compute_accuracy(self, dataset, sents_tags):
        total = 0
        correct = 0

        for (sent, pred_sent_tags) in zip(dataset, sents_tags):
            for ([_, tag], pred_tag) in zip(sent, pred_sent_tags):
                total += 1
                if tag == pred_tag: correct += 1

        return correct / total

    def _create_model(self):
        word_embed_dims = self._knobs.get('word_embed_dims')
        word_rnn_hidden_size = self._knobs.get('word_rnn_hidden_size')
        word_dropout = self._knobs.get('word_dropout')
        lr = self._knobs.get('learning_rate')

        word_count = len(self._word_dict)
        net =  PyNet(word_count + 1, self._tag_count + 1, \
                word_embed_dims, word_rnn_hidden_size, word_dropout)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        return (net, optimizer)

class PyNet(nn.Module):
    def __init__(self, word_count, tag_count, word_embed_dims=8, \
                word_rnn_hidden_size=64, word_dropout=0.2):
        super(PyNet, self).__init__()

        # Properties
        self._Ew = word_embed_dims
        self._t = tag_count
        self._h = word_rnn_hidden_size

        # Layers
        self._word_embed = nn.Embedding(word_count, self._Ew)
        self._word_lstm = nn.LSTM(self._Ew, self._h, batch_first=True, bidirectional=True)
        self._word_lin = nn.Linear(2 * self._h, self._t)
        self._word_dropout = nn.Dropout(p=word_dropout)
        
    def forward(self, words_tsr):
        N = words_tsr.size(0) # Batch size
        W = words_tsr.size(1) # No. of words per sentence
        Ew = self._Ew
        t = self._t 
        h = self._h

        # Compute word embed for each word for all sentences (N x W x Ew)
        words_embed_tsr = self._word_embed(words_tsr.view(-1)).view(N, W, Ew)

        # Apply dropout to word rep (N x W x Ew)
        words_rep_tsr = self._word_dropout(words_embed_tsr)

        # Apply bidirectional LSTM to word rep sequence (N x W x 2h)
        (words_hidden_rep_tsr, _) = self._word_lstm(words_rep_tsr)
        words_hidden_rep_tsr = words_hidden_rep_tsr.contiguous()

        # Apply linear + softmax operation for sentence rep for all sentences (N x W x t)
        word_probs_tsr = F.softmax(self._word_lin(words_hidden_rep_tsr.view(N * W, self._h * 2)), dim=1).view(N, W, t)
        
        return word_probs_tsr

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='PyBiLstm',
        task=TaskType.POS_TAGGING,
        dependencies={
            ModelDependency.PYTORCH: '0.4.1'
        },
        train_dataset_uri='data/ptb_for_pos_tagging_train.zip',
        test_dataset_uri='data/ptb_for_pos_tagging_test.zip',
        queries=[
            ['Ms.', 'Haag', 'plays', 'Elianti', '18', '.'],
            ['The', 'luxury', 'auto', 'maker', 'last', 'year', 'sold', '1,214', 'cars', 'in', 'the', 'U.S.']
        ]
    )
    
    