import torch
from egnn_pytorch import EGNN as EGNNLayer
from egnn_pytorch.egnn_pytorch import SiLU, CoorsNorm
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


class ThinEGNNLayer(EGNNLayer):
    def __init__(
            self,
            dim,
            edge_dim=0,
            m_dim=16,
            fourier_features=0,
            num_nearest_neighbors=0,
            dropout=0.0,
            init_eps=1e-3,
            norm_feats=False,
            norm_coors=False,
            update_feats=True,
            update_coors=True,
            only_sparse_neighbors=False,
            valid_radius=float('inf'),
            m_pool_method='sum',
    ):
        super(EGNNLayer, self).__init__()
        assert m_pool_method in {'sum', 'mean'}, \
            'pool method must be either sum or mean'
        assert update_feats or update_coors, \
            'you must update either features, coordinates, or both'

        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        dropout_fn = lambda: nn.Dropout(
            dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, m_dim),
            dropout_fn(),
            SiLU(),
        )

        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm() if norm_coors else nn.Identity()

        self.m_pool_method = m_pool_method

        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim),
            dropout_fn(),
            SiLU(),
        ) if update_feats else None

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, 1),
            dropout_fn(),
            SiLU(),
        ) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.init_eps = init_eps
        self.apply(self.init_)


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
                  dropout=0.0, num_layers=6, fourier_features=16,
                  norm_coords=True, norm_feats=False, thin_mlps=False,
                  **kwargs):
        m_dim = 12
        layer_class = ThinEGNNLayer if thin_mlps else EGNNLayer
        egnn = lambda: layer_class(
            dim=k, m_dim=m_dim, norm_coors=norm_coords, norm_feats=norm_feats,
            dropout=dropout, fourier_features=fourier_features,
            num_nearest_neighbors=nbhd, init_eps=5e-4)

        return nn.Sequential(
            Pass(nn.Linear(dim_input, k), dim=1),
            *[EGNNPass(egnn()) for _ in range(num_layers)],
            GlobalPool(mean=True),
            nn.Linear(k, dim_output)
        )

    def forward(self, x):
        return self.layers(x)

    def _get_min_max(self):
        """For debugging: print min and max value in each layer."""
        for i in range(3):
            print()
        network = self.layers
        res = ''
        min_ = 1e7
        max_ = -1e7
        min_abs = 1e7
        for layer in network:
            if isinstance(layer, nn.Linear):
                min_ = min(min_, float(torch.min(layer.weight)))
                max_ = max(max_, float(torch.max(layer.weight)))
                min_abs = min(min_abs,
                              float(torch.min(torch.abs(layer.weight))))
                res += 'Linear: {0}, {1} {2}\n'.format(
                    float(torch.min(layer.weight)),
                    float(torch.max(layer.weight)),
                    float(torch.min(torch.abs(layer.weight))))
            if isinstance(layer, Pass):
                if isinstance(layer.module, nn.Linear):
                    min_ = min(min_, float(torch.min(layer.module.weight)))
                    max_ = max(max_, float(torch.max(layer.module.weight)))
                    min_abs = min(min_abs, float(
                        torch.min(torch.abs(layer.module.weight))))
                    res += 'Linear:{0} {1} {2}\n'.format(
                        float(torch.min(layer.module.weight)),
                        float(torch.max(layer.module.weight)),
                        float(torch.min(torch.abs(layer.module.weight))))
            elif isinstance(layer, EGNNPass):
                layer = layer.egnn
                for network_type, network_name in zip(
                        (layer.edge_mlp, layer.node_mlp, layer.coors_mlp),
                        ('EGNN-edge', 'EGNN-node', 'EGNN-coors')):
                    for sublayer in network_type:
                        if isinstance(sublayer, nn.Linear):
                            max_ = max(max_, float(torch.max(sublayer.weight)))
                            min_ = min(min_, float(torch.min(sublayer.weight)))
                            min_abs = min(min_abs, float(
                                torch.min(torch.abs(sublayer.weight))))
                            res += network_name + ': {0} {1} {2}\n'.format(
                                float(torch.min(sublayer.weight)),
                                float(torch.max(sublayer.weight)),
                                float(torch.min(torch.abs(sublayer.weight))))
        return res[:-1], min_, max_, min_abs
