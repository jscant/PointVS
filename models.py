from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lie_conv.lieConv import LieConv, Swish, GlobalPool
from lie_conv.lieGroups import SE3
from lie_conv.masked_batchnorm import MaskBatchNormNd
from lie_conv.utils import Pass, Expression


# modified from CNN of Imrie et al. 2019
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate,
                 conv, fill):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1', MaskBatchNormNd(num_input_features))
        self.add_module('relu1', Pass(nn.ReLU(), dim=1))
        self.add_module('conv1', conv(num_input_features, growth_rate, fill))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        with open(Path('~/Desktop/shapes.txt').expanduser(), 'a') as f:
            f.write(str(x[1].shape) + '\n' + str(new_features[1].shape) + '\n')
        if isinstance(x, tuple):
            x = x[1]
        if isinstance(new_features, tuple):
            new_features = new_features[1]
        return torch.cat([x, new_features], dim=2)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate,
                 conv, fill):
        super(_DenseBlock, self).__init__()
        self.layers = []
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, drop_rate, conv, fill)
            self.layers.append(layer)

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features, conv, fill):
        super(_Transition, self).__init__()

        self.layers = []
        # self.add_module('norm', MaskBatchNormNd(num_input_features))
        self.layers.append(Pass(nn.ReLU(), dim=1))
        self.layers.append(conv(num_input_features, num_output_features, fill))
        self.layers.append(Pass(nn.MaxPool3d(2, 2), dim=1))

        def forward(self, x):
            x = self.layers[0](x)
            for layer in self.layers[1:]:
                x = layer(x)
            return x


class DenseNet(nn.Sequential):
    """DenseNet model class
    Args:
        dims - dimensions of the input image (channels, x_dim, y_dim, z_dim)
        growth_rate (int) - how many filters to add each layer (k in DenseNet
            paper)
        block_config (list of 3 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
            convolution layer
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        last_fc (bool) - include classifier layer
    """

    def __init__(self, input_c, growth_rate=16, block_config=(4, 4, 4),
                 num_init_features=32, drop_rate=0, num_classes=2,
                 last_fc=True, nbhd=20, ds_frac=1.0, bn=True, act='swish',
                 mean=True, group=SE3(), cache=False, knn=True, fill=0.1,
                 **kwargs):

        conv = lambda ki, ko, fill: LieConv(
            ki, ko, mc_samples=nbhd, ds_frac=ds_frac, bn=bn, act=act, mean=mean,
            group=group, fill=fill, cache=cache, knn=knn, **kwargs)

        super().__init__(**kwargs)
        self.last_fc = last_fc

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', conv(input_c, num_init_features, fill))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                growth_rate=growth_rate, drop_rate=drop_rate,
                                conv=conv, fill=fill)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features,
                                    conv=conv, fill=fill)
                self.features.add_module('transition%d' % (i + 1), trans)

        # Final batch norm + relu
        # self.features.add_module(
        #    'norm4', MaskBatchNormNd(255) if bn else nn.Sequential())
        self.features.add_module('relu4', Pass(nn.ReLU(), dim=1))
        self.features.add_module('globalmaxpool', GlobalPool(mean=mean))

        self.final_num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)


# define network architecture
class LieDenseNet(nn.Module):
    def __init__(self, chin, ds_frac=1, nbhd=np.inf,
                 act='swish', bn=True, num_layers=6, mean=True,
                 pool=True, liftsamples=1, fill=0.1, group=SE3(), knn=False,
                 cache=False, **kwargs):
        super().__init__()

        self.net = DenseNet(
            chin, growth_rate=kwargs.get('growth_rate', 16),
            block_config=kwargs.get('block_config', (4, 4, 4)),
            num_init_features=kwargs.get('num_init_features', 32), drop_rate=0,
            num_classes=2, last_fc=True, nbhd=nbhd, ds_frac=ds_frac, bn=True,
            act=act, mean=mean, group=group, cache=cache, knn=knn, fill=fill
        )

        self.liftsamples = liftsamples
        self.group = group

    def forward(self, x):
        lifted_x = self.group.lift(x, self.liftsamples)
        return self.net(lifted_x)


# define network architecture
class GninaNet(nn.Module):
    def __init__(self, chin, ds_frac=1, num_outputs=1, k=1536, nbhd=np.inf,
                 act='swish', bn=True, num_layers=6, mean=True,
                 pool=True, liftsamples=1, fill=0.1, group=SE3(), knn=False,
                 cache=False, **kwargs):
        super().__init__()
        if isinstance(fill, (float, int)):
            fill = [fill] * num_layers
        if isinstance(k, int):
            k = [k] * (num_layers + 1)
        conv = lambda ki, ko, fill: LieConv(
            ki, ko, mc_samples=nbhd, ds_frac=ds_frac, bn=bn, act=act, mean=mean,
            group=group, fill=fill, cache=cache, knn=knn, **kwargs)

        self.net = nn.Sequential(
            Pass(nn.Linear(chin, k[0]), dim=1),  # embedding layer
            *[conv(k[i], k[i + 1], fill[i]) for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(Swish() if act == 'swish' else nn.ReLU(), dim=1),
            Pass(nn.Linear(k[-1], num_outputs), dim=1),
            Pass(nn.Sigmoid(), dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]), )

        self.liftsamples = liftsamples
        self.group = group

    def forward(self, x):
        lifted_x = self.group.lift(x, self.liftsamples)
        return self.net(lifted_x)
