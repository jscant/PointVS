import numpy as np
import torch
from egnn_pytorch import EGNN
from egnn_pytorch.egnn_pytorch import fourier_encode_dist, exists
from einops import rearrange, repeat
from eqv_transformer.utils import Swish
from lie_conv.masked_batchnorm import MaskBatchNormNd
from lie_conv.utils import Pass
from torch import nn, einsum

from point_vs.models.layers import EGNNBatchNorm, EGNNGlobalPool
from point_vs.models.point_neural_network import PointNeuralNetwork


class EnTransformerBlock(EGNN):
    def forward(self, x):
        if len(x) == 3:
            coors, feats, mask = x
            edges = None
        else:
            coors, feats, mask, edges = x
        b, n, d, fourier_features = *feats.shape, self.fourier_features

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(
            coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist,
                                           num_encodings=fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        feats_i = repeat(feats, 'b i d -> b i n d', n=n)
        feats_j = repeat(feats, 'b j d -> b n j d', n=n)
        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        coors_out = einsum('b i j, b i j c -> b i c', coor_weights,
                           rel_coors) + coors

        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return coors_out, node_out, mask


class EnResBlock(nn.Module):

    def __init__(self, chin, chout, conv, bn=False, act='swish'):
        super().__init__()
        nonlinearity = Swish if act == 'swish' else nn.ReLU
        self.conv = conv
        self.net = nn.ModuleList([
            MaskBatchNormNd(chin) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            Pass(nn.Linear(chin, chin // 4), dim=1),
            MaskBatchNormNd(chin // 4) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            self.conv,
            MaskBatchNormNd(chout // 4) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            Pass(nn.Linear(chout // 4, chout), dim=1),
        ])
        self.chin = chin

    def forward(self, inp):
        sub_coords, sub_values, mask = inp
        for layer in self.net:
            inp = layer(inp)
        new_coords, new_values, mask = inp
        new_values[..., :self.chin] += sub_values
        return new_coords, new_values, mask


class EGNNStack(PointNeuralNetwork):

    @staticmethod
    def xavier_init(m):
        pass

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return tuple([ten.cuda() for ten in x])

    def build_net(self, chin, dim_output=2, k=12, act="swish", bn=True,
                  dropout=0.0, num_layers=6, mean=False, pool=True, feats_idx=0,
                  **kwargs):
        egnn = lambda: EGNN(dim=chin, m_dim=k, norm_rel_coors=True,
                            norm_coor_weights=False, dropout=dropout)
        if bn:
            bn = lambda: EGNNBatchNorm(12)
            eggn_layers = [(egnn(), bn()) for _ in range(num_layers)]
        else:
            eggn_layers = [(egnn(),) for _ in range(num_layers)]
        if act == 'swish':
            activation_class = Swish
        elif act == 'relu':
            activation_class = nn.ReLU
        else:
            raise NotImplementedError('{} not a recognised activation'.format(
                act))
        self.layers = nn.ModuleList([
            *[a for b in eggn_layers for a in b],
            Pass(nn.Linear(chin, chin * 2), dim=feats_idx),
            Pass(activation_class(), dim=feats_idx),
            Pass(nn.Linear(chin * 2, chin), dim=feats_idx),
            EGNNGlobalPool(
                dim=feats_idx, tensor_dim=1,
                mean=mean) if pool else nn.Sequential(),
            Pass(nn.Linear(chin, chin * 2), dim=feats_idx),
            Pass(activation_class(), dim=feats_idx),
            Pass(nn.Linear(chin * 2, dim_output), dim=feats_idx),
        ])

    def forward(self, x):
        coords, feats, mask = x
        for layer in self.layers:
            if isinstance(layer, EGNN):
                feats, coords = layer(feats, coords, mask=mask)
            else:
                x = layer([feats, coords])
                if isinstance(x, (tuple, list)):
                    feats, coords = x
                else:
                    feats = x
        return feats

    @staticmethod
    def get_min_max(network):
        min_val, max_val = np.inf, -np.inf
        for layer in network:
            if isinstance(layer, nn.Linear):
                min_val = min(float(torch.min(layer.weight)), min_val)
                max_val = max(float(torch.max(layer.weight)), max_val)
        return min_val, max_val
