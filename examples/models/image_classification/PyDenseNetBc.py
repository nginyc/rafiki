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

import math
import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint as cp
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from collections import namedtuple
from collections import OrderedDict
import argparse

from rafiki.constants import ModelDependency
from rafiki.model import BaseModel, utils, FixedKnob, FloatKnob, CategoricalKnob, PolicyKnob
from rafiki.model.dev import test_model_class

_Model = namedtuple('_Model', ['net', 'step'])

class PyDenseNetBc(BaseModel):
    '''
        Implements DenseNet-BC of "Densely Connected Convolutional Networks" for IMAGE_CLASSIFICATION,
        configured for *hyperparameter tuning with distributed parameter sharing* on Rafiki.
        
        Original paper: https://arxiv.org/abs/1608.06993
        Implementation is with credits to https://github.com/gpleiss/efficient_densenet_pytorch
    '''
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self._model = None

    @staticmethod
    def get_knob_config():
        return {
            'trial_epochs': FixedKnob(300),
            'lr': FloatKnob(1e-4, 1, is_exp=True),
            'lr_decay': FloatKnob(1e-3, 1e-1, is_exp=True),
            'opt_momentum': FloatKnob(0.7, 1, is_exp=True),
            'opt_weight_decay': FloatKnob(1e-5, 1e-3, is_exp=True),
            'batch_size': CategoricalKnob([32, 64, 128]),
            'drop_rate': FloatKnob(0, 0.4),
            'max_image_size': FixedKnob(32),
            'share_params': PolicyKnob('SHARE_PARAMS'),

            # Affects whether training is shortened by using early stopping
            'quick_train': PolicyKnob('EARLY_STOP'), 
            'early_stop_train_val_samples': FixedKnob(1024),
            'early_stop_patience_epochs': FixedKnob(5)
        }

    def train(self, dataset_path, shared_params=None):
        (train_dataset, train_val_dataset, self._train_params) = self._load_train_dataset(dataset_path)
        self._model = self._build_model()
        if self._knobs['share_params'] and shared_params is not None:
            utils.logger.log('Loading shared parameters...')
            self._model = self._load_model_parameters(self._model, shared_params)
        self._model = self._train_model(self._model, train_dataset, train_val_dataset)

    def evaluate(self, dataset_path):
        dataset = self._load_val_dataset(dataset_path, self._train_params)
        (_, acc) = self._predict(dataset)
        return acc

    def predict(self, queries):
        dataset = self._load_predict_dataset(queries, self._train_params)
        (probs, _) = self._predict(dataset)
        return probs

    def dump_parameters(self):
        (net, step) = self._model

        net_params = self._state_dict_to_params(net.state_dict())
        net_params = self._namespace_params(net_params, 'net')
        
        params = {
            **net_params,
            'train_params': json.dumps(self._train_params),
            'step': step
        }

        return params

    def load_parameters(self, params):
        self._train_params = json.loads(params['train_params'])
        model = self._build_model()
        self._model = self._load_model_parameters(model, params)

    ####################################
    # Private methods
    ####################################

    def _load_model_parameters(self, model, params):
        step = params['step']
        net_params = self._extract_namespace_from_params(params, 'net')
        net_state_dict = self._params_to_state_dict(net_params)
        net = model.net
        net.load_state_dict(net_state_dict, strict=False)
        model = _Model(net, step)
        return model

    def _predict(self, dataset):
        batch_size = self._knobs['batch_size']
        N = len(dataset)
        net = self._model.net

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        net.eval()
        [net] = self._set_device([net])

        corrects = 0
        probs = []
        with torch.no_grad():
            for (batch_images, batch_classes) in dataloader: 
                [batch_images, batch_classes] = self._set_device([batch_images, batch_classes])
                batch_probs = net(batch_images)
                probs.extend(batch_probs.cpu().tolist())
                batch_preds = batch_probs.max(1)[1]
                corrects += sum(batch_preds.eq(batch_classes).cpu().numpy())
        
        probs = self._softmax(probs)
        acc = corrects / N

        return (probs, acc)
    
    def _build_model(self):
        drop_rate = self._knobs['drop_rate']
        K = self._train_params['K']

        utils.logger.log('Building model...')

        net = DenseNet(num_classes=K, drop_rate=drop_rate)
        self._count_model_parameters(net)

        return _Model(net, 0)

    def _train_model(self, model, train_dataset, train_val_dataset):
        trial_epochs = self._knobs['trial_epochs']
        batch_size = self._knobs['batch_size']
        early_stop_patience = self._knobs['early_stop_patience_epochs']
        quick_train = self._knobs['quick_train']
        (net, step) = model

        # Define plots
        utils.logger.define_plot('Losses over Epoch', ['train_loss', 'train_val_loss'], x_axis='epoch')
        utils.logger.define_plot('Accuracies over Epoch', ['train_acc', 'train_val_acc'], x_axis='epoch')

        utils.logger.log('Training model...')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_val_dataloader = DataLoader(train_val_dataset, batch_size=batch_size)
        (optimizer, scheduler) = self._get_optimizer(net, trial_epochs)

        net.train()
        [net] = self._set_device([net])

        early_stop_condition = EarlyStopCondition(patience=early_stop_patience)
        for epoch in range(trial_epochs):
            utils.logger.log('Running epoch {}...'.format(epoch))

            scheduler.step()
            
            # Run through train dataset
            train_loss = RunningAverage()
            train_acc = RunningAverage()
            for (batch_images, batch_classes) in train_dataloader:
                [batch_images, batch_classes] = self._set_device([batch_images, batch_classes])
                probs = net(batch_images)
                loss = F.cross_entropy(probs, batch_classes)
                preds = probs.max(1)[1]
                acc = np.mean(preds.eq(batch_classes).cpu().numpy()) 
                step += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss.add(loss.item())
                train_acc.add(acc)

            utils.logger.log(epoch=epoch, step=step, 
                            train_loss=train_loss.get(), train_acc=train_acc.get())
            
            # Run through train-val dataset, if exists
            if len(train_val_dataset) > 0:
                train_val_loss = RunningAverage()
                train_val_acc = RunningAverage()
                for (batch_images, batch_classes) in train_val_dataloader:
                    [batch_images, batch_classes] = self._set_device([batch_images, batch_classes])
                    probs = net(batch_images)
                    loss = F.cross_entropy(probs, batch_classes)
                    preds = probs.max(1)[1]
                    acc = np.mean(preds.eq(batch_classes).cpu().numpy()) 
                    train_val_loss.add(loss.item())
                    train_val_acc.add(acc)

                utils.logger.log(epoch=epoch, train_val_loss=train_val_loss.get(), 
                                train_val_acc=train_val_acc.get())

                # Early stop on train-val batch loss
                if early_stop_condition.check(train_val_loss.get()):
                    utils.logger.log('Average train-val batch loss has not improved for {} epochs'.format(early_stop_condition.patience))
                    utils.logger.log('Early stopping...')
                    break

        model = _Model(net, step)
        return model

    def _load_train_dataset(self, dataset_path):
        early_stop_train_val_samples = self._knobs['early_stop_train_val_samples']
        max_image_size = self._knobs['max_image_size']
        quick_train = self._knobs['quick_train']

        # Allocate train val only if early stopping
        train_val_samples = early_stop_train_val_samples if quick_train else 0

        utils.logger.log('Loading train dataset...')

        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=max_image_size, 
                                                        mode='RGB', if_shuffle=True)
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        train_val_samples = min(dataset.size // 5, train_val_samples) # up to 1/5 of samples for train-val
        (train_images, train_classes) = (images[train_val_samples:], classes[train_val_samples:])
        (train_val_images, train_val_classes) = (images[:train_val_samples], classes[:train_val_samples])

        # Compute normalization params from train data
        norm_mean = np.mean(np.asarray(train_images) / 255, axis=(0, 1, 2)).tolist()
        norm_std = np.std(np.asarray(train_images) / 255, axis=(0, 1, 2)).tolist()

        train_dataset = ImageDataset(train_images, train_classes, dataset.image_size, 
                                    norm_mean, norm_std, is_train=True)
        train_val_dataset = ImageDataset(train_val_images, train_val_classes, dataset.image_size, 
                                        norm_mean, norm_std, is_train=False)
        train_params = {
            'norm_mean': norm_mean,
            'norm_std': norm_std,
            'image_size': dataset.image_size,
            'N': dataset.size,
            'K': dataset.classes
        }

        utils.logger.log('Train dataset has {} samples'.format(len(train_dataset)))
        utils.logger.log('Train-val dataset has {} samples'.format(len(train_val_dataset)))
        
        return (train_dataset, train_val_dataset, train_params)

    def _load_val_dataset(self, dataset_path, train_params):
        image_size = train_params['image_size']
        norm_mean = train_params['norm_mean']
        norm_std = train_params['norm_std']

        utils.logger.log('Loading val dataset...')

        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, max_image_size=image_size, 
                                                        mode='RGB')
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        val_dataset = ImageDataset(images, classes, dataset.image_size, 
                                    norm_mean, norm_std, is_train=False)
        return val_dataset

    def _load_predict_dataset(self, images, train_params):
        image_size = train_params['image_size']
        norm_mean = train_params['norm_mean']
        norm_std = train_params['norm_std']

        images = utils.dataset.transform_images(images, image_size=image_size, mode='RGB')
        classes = [0 for _ in range(len(images))]
        dataset = ImageDataset(images, classes, image_size, norm_mean, norm_std, is_train=False)
        return dataset

    def _get_optimizer(self, net, trial_epochs):
        lr = self._knobs['lr']
        lr_decay = self._knobs['lr_decay']
        opt_weight_decay = self._knobs['opt_weight_decay']
        opt_momentum = self._knobs['opt_momentum']

        optimizer = optim.SGD(net.parameters(), lr=lr, nesterov=True, 
                            momentum=opt_momentum, weight_decay=opt_weight_decay)   
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * trial_epochs, 0.75 * trial_epochs],
                            gamma=lr_decay)

        return (optimizer, scheduler)

    def _count_model_parameters(self, net):
        params_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        utils.logger.log('Model has {} parameters'.format(params_count))
        return params_count

    def _set_device(self, tensors):
        if torch.cuda.is_available():
            return [x.cuda() for x in tensors]
        else:            
            return tensors

    def _softmax(self, nums):
        nums_exp = np.exp(nums)
        sums = np.sum(nums_exp, axis=1)
        nums = (nums_exp.transpose() / sums).transpose()
        return nums.tolist()

    def _state_dict_to_params(self, state_dict):
        params = {}
        # For each tensor, convert into numpy array
        for (name, value) in state_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            else:
                raise Exception(f'Param not supported: {value}')

            params[name] = value
              
        return params 

    def _params_to_state_dict(self, params):
        state_dict = {}
        # For each tensor, convert into numpy array
        for (name, value) in params.items():
            state_dict[name] = torch.from_numpy(value)
                
        return state_dict 

    def _namespace_params(self, params, namespace):
        # For each param, add namespace prefix
        out_params = {}
        for (name, value) in params.items():
            out_params[f'{namespace}:{name}'] = value
        
        return out_params

    def _extract_namespace_from_params(self, params, namespace):
        out_params = {}
        # For each param, check for matching namespace, adding to out params without namespace prefix if matching 
        for (name, value) in params.items():
            if name.startswith(f'{namespace}:'):
                param_name = name[(len(namespace)+1):]
                out_params[param_name] = value
        
        return out_params

#####################################################################################
# Implementation of DenseNet
#####################################################################################

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

#####################################################################################
# Utils
#####################################################################################

class ImageDataset(Dataset):
    def __init__(self, images, classes, image_size, norm_mean, norm_std, is_train=False):
        self._images = images
        self._classes = classes
        if is_train:
            self._transform = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else:
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        image_class =  self._classes[idx]

        image_class = torch.tensor(image_class)
        if self._transform:
            image = self._transform(Image.fromarray(image))
        else:
            image = torch.tensor(image)

        return (image, image_class)

class RunningAverage():
    def __init__(self):
        self._avg = 0
        self._count = 0
            
    def add(self, val):
        self._avg = self._avg * self._count / (self._count + 1) + val / (self._count + 1)
        self._count += 1
        
    def get(self) -> float:
        return self._avg

class TimedRepeatCondition():
    def __init__(self, every_secs=60):
        self._every_secs = every_secs
        self._last_trigger_time = datetime.now()
            
    def check(self) -> bool:
        if (datetime.now() - self._last_trigger_time).total_seconds() >= self._every_secs:
            self._last_trigger_time = datetime.now()
            return True
        else:
            return False

class EarlyStopCondition():
    '''
    :param int patience: How many steps should the condition tolerate before calling early stop (-1 for no stop)
    '''
    def __init__(self, patience=5, if_max=False):
        self._patience = patience
        self._if_max = if_max
        self._last_best = float('inf') if not if_max else float('-inf')
        self._wait_count = 0

    @property
    def patience(self):
        return self._patience
    
    # Returns whether should early stop
    def check(self, value) -> bool:        
        if self._patience < 0: # No stop
            return False

        if (not self._if_max and value < self._last_best) or \
            (self._if_max and value > self._last_best):
            self._wait_count = 0
            self._last_best = value
        else:
            self._wait_count += 1

        if self._wait_count >= self._patience:
            return True
        else:
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/cifar10_train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/cifar10_val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/cifar10_test.zip', help='Path to test dataset')
    parser.add_argument('--query_path', type=str, default='examples/data/image_classification/cifar10_test_1.png', 
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    test_model_class(
        model_file_path=__file__,
        model_class='PyDenseNetBc',
        task='IMAGE_CLASSIFICATION',
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            ModelDependency.TORCHVISION: '0.2.2'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )