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
import gc
import json
import tempfile
import base64
import pickle
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_
from torch.utils.data import Dataset, DataLoader

from rafiki.model import BaseModel, IntegerKnob, CategoricalKnob, FloatKnob, utils
from rafiki.model.dev import test_model_class
from rafiki.constants import ModelDependency

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pandas.api.types import is_numeric_dtype
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
warnings.simplefilter(action='ignore', category=FutureWarning)

''' 
    This model is desigined for Home Credit Default Risk `https://www.kaggle
    .com/c/home-credit-default-risk` and only uses the main table 
    'application_{train|test}.csv' of this competition as dataset. Some parts of
    preprocessing is based on old version `fastai` module, which is no longer
    supported and could cause imcompatible of other modules if forcibly used.
    So a part of 'fastai.structured' is embeded in this file to ensure the model
    runs well.
'''

##########################
# fastai.structured module
##########################

def get_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

# for numeric data, fill up with median values
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

# for category data, convert to indexes, and plus 1
# the NA was remarked as -1 and then become 0
def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or col.nunique()>max_n_cat):
        df[name] = col.cat.codes+1


def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    df = df.copy()
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res


##########################################
# pytorch dataset for the home credit data
##########################################

class HomeCreditData(Dataset):
    def __init__(self, source, target, cat_list):
        self.x_cat = source[cat_list].values.tolist()
        self.x_cont = source.drop(cat_list, axis=1).values.tolist()
        self.y = None if target is None else target.values.tolist()
        self.size = source.shape[0]

    def __getitem__(self, index):
        if self.y is not None:
            return torch.LongTensor(self.x_cat[index]), \
                   torch.Tensor(self.x_cont[index]), \
                   self.y[index]
        else:
            return torch.LongTensor(self.x_cat[index]), \
                   torch.Tensor(self.x_cont[index])

    def __len__(self):
        return self.size


####################################
# This modle receiving mixed inputs
# category and continuous data type
####################################

class MixedInputModel(nn.Module):
    def __init__(self, emb_sizes, n_cont, emb_drop, out_size, sizes, drops, \
                 y_range=None, use_bn=False, is_reg=True, is_multi=False):
        super().__init__()

        self.embs = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_sizes])
        for emb in self.embs:
            x = emb.weight.data
            sc = 2 / (x.size(1) + 1)
            x.uniform_(-sc, sc)

        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont

        sizes = [n_emb + n_cont] + sizes
        self.linears = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
        for o in self.linears:
            kaiming_normal_(o.weight.data)

        self.bns = nn.ModuleList([nn.BatchNorm1d(s) for s in sizes[1:]])
        self.outp = nn.Linear(sizes[-1], out_size)
        kaiming_normal_(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn, self.y_range = use_bn, y_range
        self.is_reg = is_reg
        self.is_multi = is_multi

    def forward(self, x_cat, x_cont):
        x = []
        for i, e in enumerate(self.embs):
            x.append(e(x_cat[:, i]))
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn(x_cont)
        x = torch.cat([x, x2], 1)
        for l, d, b in zip(self.linears, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x = self.outp(x)
        x = F.log_softmax(x, dim=1)
        return x


class DNNTorch(BaseModel):
    '''
    Implements a Multi-input neural network for tabular data classification task
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epoch': IntegerKnob(5, 10),
            'learning_rate': FloatKnob(1e-3, 1e-1, is_exp=True),
            'layer_dim': CategoricalKnob([50, 100, 250])
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)

    def train(self, dataset_url, features=None, target=None, exclude=None, **kwargs):

        utils.logger.define_plot(
            'Loss Over Epochs',
            ['train_loss', 'validate_loss', 'roc_auc_score'],
            x_axis='epoch')

        # Record features & target
        self._features = features
        self._target = target

        # Load CSV file as pandas dataframe
        df = pd.read_csv(dataset_url, index_col=0)
        if exclude and set(df.columns.tolist()).intersection(set(exclude)) == set(exclude):
            df = df.drop(exclude, axis=1)

        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        df = df[df['CODE_GENDER'] != 'XNA']

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(df)
        # Encode categorical features
        X = self._encoding_categorical_type(X)
        # other preprocessing
        df_train = self._preprocessing(X)

        # build model, set optimizer and loss function
        self._build_model(self.layer_dim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(device)
        criterion = F.nll_loss
        optimizer = torch.optim.SGD(self._model.parameters(),
                                    lr=self.learning_rate, momentum=0.9)

        # prepare train/valid dataset
        cat_list = [n for n in self._encoding_dict]
        x_train, x_valid, y_train, y_valid = train_test_split(df_train, y,
                                    test_size=0.2, random_state=23, stratify=y)
        tarin_loader = DataLoader(HomeCreditData(x_train, y_train, cat_list),
                                  batch_size=1000, num_workers=0, shuffle=True)
        valid_loader = DataLoader(HomeCreditData(x_valid, y_valid, cat_list),
                                  batch_size=5000, num_workers=0, shuffle=True)
        del x_train, x_valid, y_train, y_valid, df_train, y
        gc.collect()

        # start training
        for i in range(self.epoch):

            train_losses = []
            valid_losses = []
            valid_scores = []

            # training progress
            self._model.train()
            for idx, (x_cat, x_cont, y) in enumerate(tarin_loader):
                x_cat, x_cont, y = map(lambda x: x.to(device), [x_cat, x_cont, y])

                optimizer.zero_grad()
                output = self._model(x_cat, x_cont)
                loss = criterion(output, y)
                train_losses.append(loss)

                loss.backward()
                optimizer.step()
                if idx % 10 == 0:
                    utils.logger.log(
                        'Train Epoch:{}\t[{}/{}]\tLoss:{:.8f}'.format(
                            i, idx, len(tarin_loader),
                            torch.Tensor(train_losses).mean().item()))

            # validating progress
            self._model.eval()
            for idx, (x_cat, x_cont, y) in enumerate(valid_loader):
                x_cat, x_cont, y = map(lambda x: x.to(device), [x_cat, x_cont, y])

                with torch.no_grad():
                    output = self._model(x_cat, x_cont)
                    loss = criterion(output, y)

                score = roc_auc_score(y.cpu(), output[:, 1].cpu())
                valid_losses.append(loss)
                valid_scores.append(score)
                if idx % 10 == 0:
                    utils.logger.log(
                        'Valid Epoch:{}\t[{}/{}]\tLoss:{:.8f}'.format(
                            i, idx, len(valid_loader),
                            torch.Tensor(valid_losses).mean().item()))

            utils.logger.log(train_loss=train_losses[-1].item(),
                             validate_loss=valid_losses[-1].item(),
                             roc_auc_score=valid_scores[-1],
                             epoch=i)

    def evaluate(self, dataset_url):
        # Load CSV file as pandas dataframe
        df = pd.read_csv(dataset_url, index_col=0)

        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        df = df[df['CODE_GENDER'] != 'XNA']

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(df)
        # Encode categorical features
        X = self._features_mapping(X)
        # other preprocessing
        df_eval = self._preprocessing(X)

        # prepare valid dataset
        cat_list = [n for n in self._encoding_dict]
        eval_loader = DataLoader(HomeCreditData(df_eval, y, cat_list),
                                 batch_size=1000, num_workers=0, shuffle=False)
        del df_eval, y
        gc.collect()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eval_scores = []
        self._model.eval()

        for idx, (x_cat, x_cont, y) in enumerate(eval_loader):
            x_cat, x_cont, y = map(lambda x: x.to(device), [x_cat, x_cont, y])

            with torch.no_grad():
                output = self._model(x_cat, x_cont)

            score = roc_auc_score(y.cpu(), output[:, 1].cpu())
            eval_scores.append(score)

        return torch.Tensor(eval_scores).mean().item()

    def predict(self, queries):

        df = pd.DataFrame(queries)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(df, method='predict')
        # Encode categorical features
        X = self._features_mapping(X)
        # other preprocessing
        df_test = self._preprocessing(X)

        # prepare valid dataset
        cat_list = [n for n in self._encoding_dict]
        test_loader = DataLoader(HomeCreditData(df_test, y, cat_list),
                                 batch_size=1000, num_workers=0, shuffle=False)
        del df_test, y
        gc.collect()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prediction = []
        self._model.eval()

        for idx, (x_cat, x_cont) in enumerate(test_loader):
            x_cat, x_cont = map(lambda x: x.to(device), [x_cat, x_cont])

            with torch.no_grad():
                output = self._model(x_cat, x_cont)

            pred = np.exp(output[:, 1].cpu())
            prediction.extend(pred.data.tolist())
        return prediction

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}
        # Save network parameters
        with tempfile.NamedTemporaryFile() as tmp:
            # Save whole model to temp h5 file
            self._model.cpu()
            torch.save(self._model.state_dict(), tmp.name + '.model')
            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name + '.model', 'rb') as f:
                h5_model_bytes = f.read()

        data_config_bytes = pickle.dumps([
            self.layer_dim, self._encoding_dict, self._features, self._target,
            self._nas, self._mapper
        ])

        params['h5_model_base64'] = base64.b64encode(h5_model_bytes).decode('utf-8')
        params['data_config_base64'] = base64.b64encode(data_config_bytes).decode('utf-8')

        return params

    def load_parameters(self, params):
        # Load model parameters
        h5_model_base64 = params.get('h5_model_base64', None)
        data_config_base64 = params.get('data_config_base64', None)

        data_config_bytes = base64.b64decode(data_config_base64.encode('utf-8'))
        self.layer_dim, self._encoding_dict, self._features, self._target, \
            self._nas, self._mapper = pickle.loads(data_config_bytes)

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name + '.model', 'wb') as f:
                f.write(h5_model_bytes)

            self._build_model(self.layer_dim)
            # Load model from temp file
            self._model.load_state_dict(torch.load(tmp.name + '.model'))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(device)

    def _extract_xy(self, data, method=None):
        if self._target is None:
            self._target = 'TARGET'
        y = data[self._target] if self._target in data else None

        if self._features is None:
            X = data.drop(self._target, axis=1)
            self._features = list(X.columns)
        else:
            X = data[self._features]
        if method == 'predict':
            return (X, None)
        else:
            return (X, y)

    def _encoding_categorical_type(self, df):
        # Apply label encoding for those categorical columns
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        # deal with NaN and make index encoding
        encoding_dict = {}
        for name in cat_cols:
            df[name] = df[name].astype('category').cat.as_ordered()
            cat_codes = dict(enumerate(df[name].cat.categories))
            # make way for NaN (index:0)
            encoding_dict[name] = {cat: code + 1 for code, cat in cat_codes.items()}
        self._encoding_dict = encoding_dict
        return df

    def _preprocessing(self,df):
        # encoding, filling NaN and apply scaling on data
        nas = self._nas if hasattr(self, '_nas') else None
        mapper = self._mapper if hasattr(self, '_mapper') else None
        df_numeric, _, nas, mapper = proc_df(df, do_scale=True, na_dict=nas, mapper=mapper)

        # trim extra NA flag columns
        mapper.features = [f for f in mapper.features if f[0][0][-3:] != '_na']
        mapper.built_features = [f for f in mapper.built_features if f[0][0][-3:] != '_na']
        mapper.transformed_names_ = [f for f in mapper.transformed_names_ if f[-3:] != '_na']
        df_numeric = df_numeric.iloc[:, :len(df.columns)]

        self._nas, self._mapper = nas, mapper
        return df_numeric

    def _features_mapping(self, df):
        # Encode the categorical features with pre saved encoding dict
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        df_tmp = df.copy()
        for name in cat_cols:
            if name in self._encoding_dict:
                df_tmp[name] = df[name].map(self._encoding_dict[name]).fillna(0)
        df = df_tmp
        return df

    def _build_model(self, layer_dim):
        emb_sizes = [(len(types) + 1, round(len(types)**0.5))
                     for _, types in self._encoding_dict.items()]
        self._model = MixedInputModel(
            emb_sizes=emb_sizes,
            n_cont=len(self._features) - len(self._encoding_dict),
            emb_drop=0.05,
            out_size=2,
            sizes=[layer_dim * 2, layer_dim, layer_dim],
            drops=[0.1, 0.1, 0.1],
            y_range=None,
            use_bn=False,
            is_reg=False,
            is_multi=False)


if __name__ == '__main__':
    curpath = os.path.join(os.environ['HOME'], 'rafiki')
    os.environ.setdefault('WORKDIR_PATH', curpath)
    os.environ.setdefault('PARAMS_DIR_PATH', os.path.join(curpath, 'params'))

    train_set_url = os.path.join(curpath, 'data', 'application_train_index.csv')
    valid_set_url = train_set_url
    test_set_url = os.path.join(curpath, 'data', 'application_test_index.csv')

    test_queries = pd.read_csv(test_set_url, index_col=0).iloc[:5]
    test_queries = json.loads(test_queries.to_json(orient='records'))

    test_model_class(
        model_file_path=__file__,
        model_class='DNNTorch',
        task='TABULAR_CLASSIFICATION',
        dependencies={
            ModelDependency.TORCH: '1.3.1',
            ModelDependency.SCIKIT_LEARN: '0.21.3',
            # 'fastai':'0.7.0'
            'sklearn-pandas': '1.8.0',
        },
        train_dataset_path=train_set_url,
        val_dataset_path=valid_set_url,
        train_args={
            'target': 'TARGET',
            'exclude': ['SK_ID_CURR'],
        },
        queries=test_queries,
    )
