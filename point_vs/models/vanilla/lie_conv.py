import numpy as np
from eqv_transformer.eqv_attention import GlobalPool
from eqv_transformer.utils import Swish
from lie_conv.lieConv import LieConv
from lie_conv.lieGroups import SE3
from lie_conv.masked_batchnorm import MaskBatchNormNd
from lie_conv.utils import Expression, Pass
from torch import nn

from point_vs.global_objects import DEVICE
from point_vs.models.vanilla.pnn_vanilla_base import PNNVanillaBase


class LieConvBottleBlock(nn.Module):
    """ A bottleneck residual block as described in figure 5"""

    def __init__(self, chin, chout, conv, bn=False, act='swish', fill=None,
                 dropout=0):
        super().__init__()
        assert chin <= chout, f"unsupported channels chin{chin}, " \
                              f"chout{chout}. No upsampling atm."
        nonlinearity = Swish if act == 'swish' else nn.ReLU
        if fill is not None:
            self.conv = conv(chin // 4, chout // 4, fill=fill)
        else:
            self.conv = conv(chin // 4, chout // 4)

        self.net = nn.Sequential(
            Pass(nonlinearity(), dim=1),
            MaskBatchNormNd(chin) if bn else nn.Sequential(),
            Pass(nn.Dropout(dropout) if dropout else nn.Sequential()),
            Pass(nn.Linear(chin, chin // 4), dim=1),
            Pass(nonlinearity(), dim=1),
            MaskBatchNormNd(chin // 4) if bn else nn.Sequential(),
            self.conv,
            Pass(nonlinearity(), dim=1),
            MaskBatchNormNd(chout // 4) if bn else nn.Sequential(),
            Pass(nn.Dropout(dropout) if dropout else nn.Sequential()),
            Pass(nn.Linear(chout // 4, chout), dim=1),
        )
        self.chin = chin

    def forward(self, inp):
        sub_coords, sub_values, mask = self.conv.subsample(inp)
        new_coords, new_values, mask = self.net(inp)
        new_values[..., :self.chin] += sub_values
        return new_coords, new_values, mask


class LieResNet(PNNVanillaBase):
    """Generic ResNet architecture from https://arxiv.org/abs/2002.12880"""

    def _get_y_true(self, y):
        return y.to(DEVICE)

    def prepare_input(self, x):
        return tuple([inp.to(DEVICE) for inp in x])

    def build_net(self, dim_input, dim_output, ds_frac=1, k=1536, nbhd=np.inf,
                  act="swish", bn=True, num_layers=6, pool=True, liftsamples=1,
                  fill=1 / 4, group=SE3, knn=False, cache=False, dropout=0,
                  **kwargs):
        """
        Arguments:
            dim_input: number of input channels: 1 for MNIST, 3 for RGB
            images, other
                for non images
            ds_frac: total downsampling to perform throughout the layers of the
                net. In (0,1)
            k: channel width for the network. Can be int (same for all) or array
                to specify individually.
            nbhd: number of samples to use for Monte Carlo estimation (p)
            act:
            bn: whether or not to use batch normalization. Recommended in al
                cases except dynamical systems.
            num_layers: number of BottleNeck Block layers in the network
            pool:
            liftsamples: number of samples to use in lifting. 1 for all groups
                with trivial stabilizer. Otherwise 2+
            fill: specifies the fraction of the input which is included in local
                neighborhood. (can be array to specify a different value for
                each layer)
            group: group to be equivariant to
            knn:
            cache:
            dropout: dropout probability for fully connected layers
        """
        if isinstance(fill, (float, int)):
            fill = [fill] * num_layers
        if isinstance(k, int):
            k = [k] * (num_layers + 1)
        conv = lambda ki, ko, fill: LieConv(
            ki, ko, mc_samples=nbhd, ds_frac=ds_frac, bn=bn, act=act, mean=True,
            group=group, fill=fill, cache=cache, knn=knn)
        layers = nn.ModuleList([
            Pass(nn.Linear(dim_input, k[0]), dim=1),
            *[LieConvBottleBlock(k[i], k[i + 1], conv, bn=bn, act=act,
                                 fill=fill[i],
                                 dropout=dropout) for i in range(num_layers)],
            Pass(nn.ReLU(), dim=1),
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(nn.Dropout(p=dropout), dim=1) if dropout else nn.Sequential(),
            GlobalPool(mean=True) if pool else Expression(lambda x: x[1]),
            nn.Linear(k[-1], dim_output)
        ])
        self.group = group
        self.liftsamples = liftsamples

        return layers

    def forward(self, x):
        x = tuple([ten.to(DEVICE) for ten in self.group.lift(x, self.liftsamples)])
        for layer in self.layers:
            x = layer(x)
        return x
