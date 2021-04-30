import torch
from egnn_pytorch import EGNN as EGNNLayer
from eqv_transformer.utils import GlobalPool
from lie_conv.utils import Pass
from torch import nn

from point_vs.models.point_neural_network import PointNeuralNetwork


class EGNNPass(nn.Module):
    def __init__(self, egnn):
        super().__init__()
        self.egnn = egnn

    def forward(self, x):
        if len(x) == 2:
            coors, feats = x
            mask = None
        else:
            coors, feats, mask = x
        feats, coors = self.egnn(feats=feats, coors=coors, mask=mask)
        return coors, feats, mask


class EGNN(PointNeuralNetwork):

    # We have our own initialisation methods for EGNN
    @staticmethod
    def xavier_init(m):
        pass

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return [i.cuda() for i in x]

    def build_net(self, dim_input, dim_output=1, k=12, nbhd=0,
                  dropout=0.0, num_layers=6, fourier_features=16, **kwargs):
        egnn = lambda: EGNNLayer(
            dim=k, m_dim=12, norm_coors=True, norm_feats=True, dropout=dropout,
            fourier_features=fourier_features, init_eps=1e-2,
            num_nearest_neighbors=nbhd)

        return nn.Sequential(
            Pass(nn.Linear(dim_input, k), dim=1),
            *[EGNNPass(egnn()) for _ in range(num_layers)],
            GlobalPool(mean=True),
            nn.Linear(k, dim_output)
        )

    def forward(self, x):
        return self.layers(x)

    def _get_min_max(self):
        network = self.layers
        for layer in network:
            if isinstance(layer, nn.Linear):
                print('Linear:',
                      float(torch.min(layer.weight)),
                      float(torch.max(layer.weight)))
            if isinstance(layer, Pass):
                if isinstance(layer.module, nn.Linear):
                    print('Linear:',
                          float(torch.min(layer.module.weight)),
                          float(torch.max(layer.module.weight)))
            elif isinstance(layer, EGNNPass):
                layer = layer.egnn
                print()
                for network_type, network_name in zip(
                        (layer.edge_mlp, layer.node_mlp, layer.coors_mlp),
                        ('EGNN-edge', 'EGNN-node', 'EGNN-coors')):
                    for sublayer in network_type:
                        if isinstance(sublayer, nn.Linear):
                            print(network_name,
                                  float(torch.min(sublayer.weight)),
                                  float(torch.max(sublayer.weight)))
