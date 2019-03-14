import math
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
from collections import OrderedDict
import abc

from rafiki.model import BaseModel, utils, FixedKnob
from rafiki.advisor import tune_model

class PyDenseNet(BaseModel):
    def __init__(self, **knobs):
        self._knobs = knobs

    @staticmethod
    def get_knob_config():
        return {
            'trial_epochs': FixedKnob(300),
            'batch_size': FixedKnob(64),
            'max_image_size': FixedKnob(28),
            'lr': FixedKnob(0.1),
            'lr_decay': FixedKnob(0.1)
        }

    def train(self, dataset_uri, shared_params):
        max_image_size = self._knobs['max_image_size']

        utils.logger.log('Loading dataset...')
        dataset = ImageDataset(dataset_uri, max_image_size, is_train=True)
        self._train_params = dataset.train_params    
        
        utils.logger.log('Training model...')
        self._net = self._train(dataset)

        utils.logger.log('Computing train accuracy...')
        acc = self._evaluate(dataset, self._net)
        utils.logger.log('Train accuracy: {}'.format(acc))

    def evaluate(self, dataset_uri):
        max_image_size = self._knobs['max_image_size']

        utils.logger.log('Loading dataset...')
        dataset = ImageDataset(dataset_uri, max_image_size, train_params=self._train_params, is_train=False)

        utils.logger.log('Computing val accuracy...')
        acc = self._evaluate(dataset, self._net)
        utils.logger.log('Val accuracy: {}'.format(acc))
        return acc

    def predict(self, queries):
        # TODO
        pass

    def save_parameters(self, params_dir):
        # TODO
        pass

    def load_parameters(self, params_dir):
        # TODO
        pass

    def _evaluate(self, dataset, net):
        batch_size = self._knobs['batch_size']
        N = len(dataset)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        net.eval()
        if torch.cuda.is_available():
            utils.logger.log('Using CUDA...')
            net = net.cuda()

        with torch.no_grad():
            # Train for epoch
            corrects = 0
            for (batch_images, batch_classes) in dataloader: 
                probs = net(batch_images)
                preds = probs.max(1)[1]
                corrects += sum(preds.eq(batch_classes).cpu().numpy())
        
            return corrects / N

    def _train(self, dataset):
        trial_epochs = self._knobs['trial_epochs']
        batch_size = self._knobs['batch_size']
        lr = self._knobs['lr']
        lr_decay = self._knobs['lr_decay']
        K = self._train_params['K']

        N = len(dataset)
        momentum = 0.9
        weight_decay = 1e-4 
        log_every_secs = 60

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        net = DenseNet(num_classes=K)
        net.train()
        if torch.cuda.is_available():
            utils.logger.log('Using CUDA...')
            net = net.cuda()

        params_count = sum([p.data.nelement() for p in net.parameters()])
        utils.logger.log('Model has {} parameters'.format(params_count))

        optimizer = optim.SGD(net.parameters(), lr=lr, nesterov=True, 
                            momentum=momentum, weight_decay=weight_decay)   
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * trial_epochs, 0.75 * trial_epochs],
                            gamma=lr_decay)

        last_log_time = datetime.now()
        step = 0
        for epoch in range(trial_epochs):
            utils.logger.log('Running epoch {}'.format(epoch))

            scheduler.step()
            
            # Train for epoch
            corrects = 0
            for (batch_images, batch_classes) in dataloader: 
                probs = net(batch_images)
                loss = F.cross_entropy(probs, batch_classes)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = probs.max(1)[1]
                step += 1
                corrects += sum(preds.eq(batch_classes).cpu().numpy())

                # Periodically, log loss
                if (datetime.now() - last_log_time).total_seconds() >= log_every_secs:
                    last_log_time = datetime.now()
                    utils.logger.log(step=step, loss=loss.item())
            
            acc = corrects / N
            utils.logger.log(epoch=epoch, acc=acc)

        return net

class ImageDataset(Dataset):
    def __init__(self, dataset_uri, max_image_size, train_params=None, is_train=True):
        if train_params is not None:
            dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=train_params['image_size'], 
                                                            mode='RGB')
            (self._images, self._classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
            norm_mean = train_params['norm_mean']
            norm_std = train_params['norm_std']
            self._train_params = train_params
        else:
            dataset = utils.dataset.load_dataset_of_image_files(dataset_uri, max_image_size=max_image_size, 
                                                                mode='RGB')
            (self._images, self._classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
            norm_mean = np.mean(np.array(self._images) / 255, axis=(0, 1, 2)).tolist() 
            norm_std = np.std(np.array(self._images) / 255, axis=(0, 1, 2)).tolist() 
            self._train_params = {
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'image_size': dataset.image_size,
                'K': dataset.classes
            }

        if is_train:
            self._transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else:
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

    @property
    def train_params(self):
        return {
            
        }

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

        if torch.cuda.is_available():
            image = image.cuda()
            image_class = image_class.cuda()

        return (image, image_class)

#####################################################################################
# Below code is with credits to https://github.com/gpleiss/efficient_densenet_pytorch
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

if __name__ == '__main__':
    tune_model(
        PyDenseNet, 
        train_dataset_uri='data/fashion_mnist_for_image_classification_train.zip',
        val_dataset_uri='data/fashion_mnist_for_image_classification_val.zip',
        test_dataset_uri='data/fashion_mnist_for_image_classification_test.zip',
        total_trials=1,
        should_save=False
    )

